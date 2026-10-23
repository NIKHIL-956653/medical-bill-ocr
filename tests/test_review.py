"""
Tests for the review UI backend: value location, accuracy metrics, storage,
the detailed pipeline, and the HTTP API. No API key required.

Run: pytest tests/ -v
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import pymupdf
import pytest

from bill_ocr import pipeline
from bill_ocr.accuracy import ReviewRecord, accuracy_report, compare_bills
from bill_ocr.errors import BillOCRError
from bill_ocr.ingest import SourceType
from bill_ocr.locate import locate
from bill_ocr.pipeline import PageView, PageWord, ProcessedDocument
from bill_ocr.schema import BillData, LineItem
from bill_ocr.store import ReviewStore
from bill_ocr.validate import validate

# words as stored: [text, confidence, x, y, w, h]
PAGE = [
    ["Test", 95.0, 0.10, 0.05, 0.08, 0.02],
    ["Clinlc", 70.0, 0.19, 0.05, 0.08, 0.02],   # OCR typo
    ["Invoice", 90.0, 0.10, 0.10, 0.10, 0.02],
    ["INV-1", 92.0, 0.21, 0.10, 0.06, 0.02],
    ["Exam", 91.0, 0.10, 0.30, 0.08, 0.02],
    ["$50.00", 88.0, 0.70, 0.30, 0.08, 0.02],
    ["Vaccine", 93.0, 0.10, 0.34, 0.08, 0.02],
    ["25.00", 90.0, 0.70, 0.34, 0.08, 0.02],
    ["Total", 94.0, 0.10, 0.40, 0.08, 0.02],
    ["$1,075.00", 85.0, 0.70, 0.40, 0.10, 0.02],
]


def make_bill(**overrides) -> BillData:
    base = dict(
        provider_name="Test Clinic", invoice_number="INV-1", bill_date="11/07/2024",
        line_items=[LineItem("Exam", 1, 50.0), LineItem("Vaccine", 1, 25.0)],
        subtotal=75.0, total=75.0, source_file="bill.png", ocr_mean_confidence=90.0,
        extraction_method="gemini",
    )
    base.update(overrides)
    return validate(BillData(**base))


def fake_processed(bill: BillData | None = None) -> ProcessedDocument:
    img = np.full((1100, 850, 3), 255, dtype=np.uint8)
    words = [PageWord(w[0], w[1], *w[2:]) for w in PAGE]
    return ProcessedDocument(bill=bill or make_bill(), source_type=SourceType.IMAGE,
                             pages=[PageView(image=img, words=words)])


# ── locate ───────────────────────────────────────────────────────────────

def test_locate_number_ignores_currency_formatting():
    boxes = locate(50, [PAGE])
    assert boxes == [{"page": 0, "x": 0.70, "y": 0.30, "w": 0.08, "h": 0.02}]
    assert len(locate("1075", [PAGE])) == 1


def test_locate_multiword_text_with_ocr_typo():
    boxes = locate("test clinic", [PAGE])
    assert [b["x"] for b in boxes] == [0.10, 0.19]


def test_locate_exact_short_token_and_misses():
    assert len(locate("INV-1", [PAGE])) == 1
    assert locate("Radiology", [PAGE]) == []
    assert locate("", [PAGE]) == []
    assert locate(None, [PAGE]) == []


def test_locate_reports_page_index():
    boxes = locate("Vaccine", [[], PAGE])
    assert boxes[0]["page"] == 1


# ── accuracy ─────────────────────────────────────────────────────────────

def test_compare_identical_bills_all_correct():
    d = make_bill().to_dict()
    assert all(r.correct for r in compare_bills(d, d))


def test_compare_detects_corrections():
    extracted = make_bill(total=57.0).to_dict()
    final = make_bill(provider_name="  test   CLINIC ").to_dict()  # whitespace/case only
    wrong = {r.field for r in compare_bills(extracted, final) if not r.correct}
    assert wrong == {"total"}


def test_compare_counts_added_line_item_as_wrong():
    extracted = make_bill().to_dict()
    final = make_bill(line_items=[LineItem("Exam", 1, 50.0), LineItem("Vaccine", 1, 25.0),
                                  LineItem("Lab", 1, 10.0)]).to_dict()
    wrong = [r.field for r in compare_bills(extracted, final) if not r.correct]
    assert "line_item_count" in wrong
    assert wrong.count("line_items.amount") == 1


def test_accuracy_report_aggregates_by_field_week_and_model():
    good = make_bill().to_dict()
    bad = make_bill(total=1.0).to_dict()
    report = accuracy_report([
        ReviewRecord(good, good, "2026-10-19T10:00:00+00:00", "gemini:a"),
        ReviewRecord(bad, good, "2026-10-20T10:00:00+00:00", "gemini:b"),
    ])
    assert report["documents"] == 2
    assert report["perfect_documents"] == 1
    total = next(f for f in report["fields"] if f["field"] == "total")
    assert total == {"field": "total", "correct": 1, "total": 2, "accuracy": 0.5}
    assert [w["week"] for w in report["by_week"]] == ["2026-W43"]
    assert {m["model"]: m["perfect_rate"] for m in report["by_model"]} == \
        {"gemini:a": 1.0, "gemini:b": 0.0}


def test_accuracy_report_empty():
    report = accuracy_report([])
    assert report["documents"] == 0 and report["field_accuracy"] is None


# ── schema + store ───────────────────────────────────────────────────────

def test_bill_from_dict_roundtrip():
    bill = make_bill(total=57.0)
    again = BillData.from_dict(bill.to_dict() | {"unknown_key": 1})
    assert again == bill


def test_store_lifecycle(tmp_path):
    store = ReviewStore(tmp_path)
    doc_id = store.create_document("../../evil name.png", b"fake")
    assert store.upload_path(doc_id).parent == tmp_path / "uploads"  # client name not used
    assert store.get_document(doc_id)["filename"] == "evil name.png"

    store.save_result(doc_id, fake_processed(make_bill(total=57.0)), "gemini:test")
    doc = store.get_document(doc_id)
    assert doc["status"] == "ready"
    assert doc["pages"][0]["words"][0][0] == "Test"
    assert store.page_path(doc_id, 0).exists()
    assert store.review_records() == []

    store.save_review(doc_id, make_bill())
    store.save_review(doc_id, make_bill(client_name="Jane"))
    assert len(store.revisions(doc_id)) == 2
    [rec] = store.review_records()
    assert rec.extracted["total"] == 57.0 and rec.final["client_name"] == "Jane"
    assert len(store.bills(reviewed_only=True)) == 1

    assert store.delete(doc_id)
    assert not store.page_path(doc_id, 0).exists()
    assert store.get_document(doc_id) is None


def test_store_restart_fails_interrupted_jobs(tmp_path):
    doc_id = ReviewStore(tmp_path).create_document("a.png", b"x")
    doc = ReviewStore(tmp_path).get_document(doc_id)  # "restart"
    assert doc["status"] == "failed" and "restart" in doc["error"]


# ── detailed pipeline (extraction stubbed out) ───────────────────────────

@pytest.fixture
def stub_extract(monkeypatch):
    seen = {}

    def fake(text, source_file=None, ocr_confidence=None):
        seen["text"] = text
        return make_bill(source_file=source_file, ocr_mean_confidence=ocr_confidence)

    monkeypatch.setattr(pipeline, "extract_fields", fake)
    return seen


def test_detailed_pipeline_digital_pdf_has_word_boxes(tmp_path, stub_extract):
    path = tmp_path / "digital.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Invoice INV-1 Total 75.00 " * 5)
    doc.save(path)
    doc.close()

    result = pipeline.process_file_detailed(path)
    assert result.source_type == SourceType.DIGITAL_PDF
    [page] = result.pages
    assert page.image.ndim == 3
    words = [w.text for w in page.words]
    assert "INV-1" in words
    assert all(0 <= w.x <= 1 and 0 <= w.y <= 1 for w in page.words)
    assert pipeline.process_file_detailed(path, with_pages=False).pages == []


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")
def test_detailed_pipeline_image_boxes_are_normalized(tmp_path, stub_extract):
    img = np.full((600, 900, 3), 255, dtype=np.uint8)
    cv2.putText(img, "TOTAL 75.00", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    path = tmp_path / "scan.png"
    cv2.imwrite(str(path), img)

    result = pipeline.process_file_detailed(path)
    [page] = result.pages
    assert page.image.shape == img.shape
    assert "TOTAL" in stub_extract["text"]
    total = next(w for w in page.words if w.text == "TOTAL")
    # text starts at x=60/900 and sits around y=300/600 on the original image
    assert abs(total.x - 60 / 900) < 0.03
    assert 0.35 < total.y < 0.55


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")
def test_scanned_pdf_no_longer_writes_rendered_pages(tmp_path, stub_extract, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img = np.full((800, 600, 3), 255, dtype=np.uint8)
    cv2.putText(img, "INVOICE", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    png = tmp_path / "page.png"
    cv2.imwrite(str(png), img)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, filename=str(png))
    doc.save(tmp_path / "scanned.pdf")
    doc.close()

    result = pipeline.process_file_detailed(tmp_path / "scanned.pdf")
    assert result.source_type == SourceType.SCANNED_PDF
    assert len(result.pages) == 1
    assert not (tmp_path / "artifacts").exists()


# ── HTTP API ─────────────────────────────────────────────────────────────

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from bill_ocr.web import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    calls = {"fail": False}

    def processor(path):
        if calls["fail"]:
            raise BillOCRError("GOOGLE_API_KEY not found")
        return fake_processed(make_bill(subtotal=None, total=57.0, source_file=path.name))

    app = create_app(tmp_path / "review", processor=processor, model_name="gemini:test")
    c = TestClient(app)
    c.calls = calls
    return c


def upload(client, name="bill.png", content=b"img"):
    r = client.post("/api/documents", files=[("files", (name, content, "image/png"))])
    assert r.status_code == 201, r.text
    return r.json()["documents"][0]["id"]


def test_api_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Medical Bill Review" in r.text


def test_api_upload_review_and_accuracy(client):
    doc_id = upload(client)
    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["status"] == "ready"
    assert doc["model"] == "gemini:test"
    assert any(f.startswith("totals_mismatch") for f in doc["current"]["validation_flags"])
    assert doc["needs_review"] is True
    assert client.get(f"/api/documents/{doc_id}/pages/0").headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/documents/{doc_id}/pages/5").status_code == 404

    boxes = client.get(f"/api/documents/{doc_id}/locate", params={"value": "50"}).json()["boxes"]
    assert len(boxes) == 1

    corrected = doc["current"] | {"total": "$75.00", "client_name": " Jane Doe ",
                                  "source_file": "spoofed.png", "ocr_mean_confidence": 100}
    r = client.put(f"/api/documents/{doc_id}", json=corrected)
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["status"] == "reviewed"
    assert saved["current"]["total"] == 75.0
    assert saved["current"]["client_name"] == "Jane Doe"
    assert saved["current"]["source_file"] == "bill.png"        # provenance not client-controlled
    assert saved["current"]["ocr_mean_confidence"] == 90.0
    assert saved["current"]["validation_flags"] == []             # re-validated after fix
    assert saved["extracted"]["total"] == 57.0                    # original kept for accuracy

    report = client.get("/api/accuracy").json()
    assert report["documents"] == 1
    wrong = [f["field"] for f in report["fields"] if f["accuracy"] < 1]
    assert sorted(wrong) == ["client_name", "total"]

    listing = client.get("/api/documents").json()["documents"]
    assert listing[0]["status"] == "reviewed" and listing[0]["needs_review"] is False
    assert len(client.get(f"/api/documents/{doc_id}/revisions").json()["revisions"]) == 1


def test_api_export(client):
    doc_id = upload(client)
    assert client.get("/api/export").status_code == 404           # nothing reviewed yet
    r = client.get("/api/export", params={"reviewed_only": False})
    assert r.status_code == 200 and r.content[:2] == b"PK"         # xlsx is a zip
    client.put(f"/api/documents/{doc_id}", json=client.get(f"/api/documents/{doc_id}").json()["current"])
    assert client.get("/api/export").status_code == 200


def test_api_rejects_bad_input(client):
    r = client.post("/api/documents", files=[("files", ("notes.docx", b"x", "application/octet-stream"))])
    assert r.status_code == 400
    r = client.post("/api/documents", files=[("files", ("empty.png", b"", "image/png"))])
    assert r.status_code == 400
    doc_id = upload(client)
    r = client.put(f"/api/documents/{doc_id}", json={"total": "twelve"})
    assert r.status_code == 422
    assert client.get("/api/documents/nope").status_code == 404


def test_api_blank_line_items_dropped(client):
    doc_id = upload(client)
    body = {"line_items": [{"description": "Exam", "amount": "75"},
                           {"description": " ", "amount": "", "quantity": None, "date": ""}],
            "subtotal": 75}
    saved = client.put(f"/api/documents/{doc_id}", json=body).json()
    assert len(saved["current"]["line_items"]) == 1


def test_api_failed_processing_and_retry(client):
    client.calls["fail"] = True
    doc_id = upload(client)
    doc = client.get(f"/api/documents/{doc_id}").json()
    assert doc["status"] == "failed" and "GOOGLE_API_KEY" in doc["error"]
    assert client.put(f"/api/documents/{doc_id}", json={}).status_code == 409

    client.calls["fail"] = False
    assert client.post(f"/api/documents/{doc_id}/reprocess").status_code == 202
    assert client.get(f"/api/documents/{doc_id}").json()["status"] == "ready"


def test_api_delete(client):
    doc_id = upload(client)
    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.get(f"/api/documents/{doc_id}").status_code == 404
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404
