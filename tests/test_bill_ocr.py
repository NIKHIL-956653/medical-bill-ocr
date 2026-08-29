"""
Automated tests for the bill OCR pipeline. No API key required.

Run: pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import pymupdf
import pytest

from bill_ocr.errors import FileReadError, UnsupportedFileError
from bill_ocr.extract import ExtractionError, _strip_fences, _to_bill_data, extract_fields
from bill_ocr.export import ExportError, bills_to_frames, export_excel
from bill_ocr.ingest import classify, SourceType
from bill_ocr.schema import BillData, LineItem
from bill_ocr.validate import needs_human_review, validate


# ── helpers: generate test files on the fly ──────────────────────────────

def make_digital_pdf(path: Path, text: str = "Invoice 123 " * 20) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


def make_image(path: Path, width: int = 400, height: int = 300) -> Path:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(img, "TEST", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.imwrite(str(path), img)
    return path


def make_bill(**overrides) -> BillData:
    base = dict(
        provider_name="Test Clinic",
        invoice_number="INV-1",
        bill_date="11/07/2024",
        line_items=[LineItem("Exam", 1, 50.0), LineItem("Vaccine", 1, 25.0)],
        subtotal=75.0,
        total=75.0,
        source_file="test.png",
        ocr_mean_confidence=90.0,
    )
    base.update(overrides)
    return BillData(**base)


# ── schema ───────────────────────────────────────────────────────────────

def test_schema_to_dict_roundtrip():
    bill = make_bill()
    d = bill.to_dict()
    assert d["provider_name"] == "Test Clinic"
    assert d["line_items"][0]["description"] == "Exam"


def test_schema_defaults_are_none_and_empty():
    bill = BillData()
    assert bill.provider_name is None
    assert bill.line_items == []
    assert bill.validation_flags == []


# ── validation ───────────────────────────────────────────────────────────

def test_validate_clean_bill_has_no_flags():
    bill = validate(make_bill())
    assert bill.validation_flags == []
    assert needs_human_review(bill) is False


def test_validate_catches_totals_mismatch():
    bill = validate(make_bill(subtotal=999.0, total=999.0))
    assert any(f.startswith("totals_mismatch") for f in bill.validation_flags)
    assert needs_human_review(bill) is True


def test_validate_tolerates_rounding():
    bill = validate(make_bill(subtotal=75.04, total=75.04))  # within 5 cents
    assert not any(f.startswith("totals_mismatch") for f in bill.validation_flags)


def test_validate_flags_low_ocr_confidence():
    bill = validate(make_bill(ocr_mean_confidence=50.0))
    assert any(f.startswith("low_ocr_confidence") for f in bill.validation_flags)


def test_validate_flags_missing_fields():
    bill = validate(make_bill(provider_name=None, invoice_number=None))
    assert "missing_provider_name" in bill.validation_flags
    assert "missing_invoice_number" in bill.validation_flags


def test_validate_flags_unparseable_date():
    bill = validate(make_bill(bill_date="not a date"))
    assert any(f.startswith("unparseable_date") for f in bill.validation_flags)


def test_low_confidence_alone_does_not_force_review():
    bill = validate(make_bill(ocr_mean_confidence=50.0))
    assert needs_human_review(bill) is False  # low conf but nothing else wrong


# ── extract helpers (no API) ─────────────────────────────────────────────

def test_strip_fences_removes_markdown():
    fenced = "```json\n{\"a\": 1}\n```"
    assert _strip_fences(fenced) == '{"a": 1}'


def test_strip_fences_leaves_clean_json_alone():
    clean = '{"a": 1}'
    assert _strip_fences(clean) == clean


def test_to_bill_data_coerces_string_numbers():
    data = {"total": "57.69", "line_items": [
        {"description": "X", "quantity": "2", "amount": "10.5"}]}
    bill = _to_bill_data(data, "f.png", 80.0)
    assert bill.total == 57.69
    assert bill.line_items[0].amount == 10.5


def test_to_bill_data_skips_garbage_items():
    data = {"line_items": [
        {"description": ""},          # empty -> skipped
        "not a dict",                  # wrong type -> skipped
        {"description": "Valid", "amount": 5}]}
    bill = _to_bill_data(data, None, None)
    assert len(bill.line_items) == 1
    assert bill.line_items[0].description == "Valid"


def test_to_bill_data_nulls_unparseable_numbers():
    data = {"total": "abc"}
    assert _to_bill_data(data, None, None).total is None


def test_extract_rejects_empty_text():
    with pytest.raises(ExtractionError):
        extract_fields("")


# ── export ───────────────────────────────────────────────────────────────

def test_export_writes_workbook_and_csvs(tmp_path):
    bills = [validate(make_bill())]
    out = export_excel(bills, tmp_path / "r.xlsx")
    assert out.exists()
    assert (tmp_path / "r_bills.csv").exists()
    assert (tmp_path / "r_line_items.csv").exists()


def test_export_review_column_marks_bad_bill():
    good = validate(make_bill())
    bad = validate(make_bill(subtotal=999.0, total=999.0, source_file="bad.png"))
    bills_df, items_df = bills_to_frames([good, bad])
    assert list(bills_df["NEEDS_REVIEW"]) == ["", "YES"]
    assert len(items_df) == 4  # 2 items per bill


def test_export_rejects_empty_list(tmp_path):
    with pytest.raises(ExportError):
        export_excel([], tmp_path / "r.xlsx")


# ── ingest router ────────────────────────────────────────────────────────

def test_classify_digital_pdf(tmp_path):
    pdf = make_digital_pdf(tmp_path / "digital.pdf")
    doc = classify(pdf)
    assert doc.source_type == SourceType.DIGITAL_PDF
    assert doc.text_chars > 50


def test_classify_image(tmp_path):
    img = make_image(tmp_path / "scan.png")
    doc = classify(img)
    assert doc.source_type == SourceType.IMAGE
    assert doc.n_pages == 1


def test_classify_rejects_unsupported_extension(tmp_path):
    f = tmp_path / "file.docx"
    f.write_text("hello")
    with pytest.raises(UnsupportedFileError):
        classify(f)


def test_classify_rejects_missing_file():
    with pytest.raises(FileReadError):
        classify("does/not/exist.pdf")


def test_classify_rejects_empty_file(tmp_path):
    f = tmp_path / "empty.pdf"
    f.touch()
    with pytest.raises(FileReadError):
        classify(f)