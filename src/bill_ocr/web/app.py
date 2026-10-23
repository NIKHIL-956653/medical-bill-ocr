"""
Review web UI: upload bills, check extracted fields against the page image,
correct them, export, and track extraction accuracy over time.

Run:  python -m bill_ocr serve            (http://127.0.0.1:8000)

The server binds to localhost by default and has no login: it is meant to run
on the reviewer's own machine, next to the bills it processes.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

from fastapi import BackgroundTasks, Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from bill_ocr import extract
from bill_ocr.accuracy import HEADER_FIELDS, ITEM_FIELDS, MONEY_FIELDS, accuracy_report
from bill_ocr.errors import BillOCRError
from bill_ocr.export import export_excel
from bill_ocr.ingest import SUPPORTED_IMAGE_EXTS
from bill_ocr.locate import locate
from bill_ocr.pipeline import ProcessedDocument, process_file_detailed
from bill_ocr.schema import BillData, LineItem
from bill_ocr.store import (STATUS_FAILED, STATUS_PROCESSING, STATUS_READY, STATUS_REVIEWED,
                            ReviewStore)
from bill_ocr.validate import needs_human_review, validate

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SUPPORTED_EXTS = {".pdf"} | SUPPORTED_IMAGE_EXTS

Processor = Callable[[Path], ProcessedDocument]


def _default_processor(path: Path) -> ProcessedDocument:
    return process_file_detailed(path)


def create_app(data_dir: str | Path = "data/review",
               processor: Processor = _default_processor,
               model_name: str | None = None) -> FastAPI:
    """
    Build the app. `processor` is injectable so tests can run without
    Tesseract or a Gemini key.
    """
    store = ReviewStore(data_dir)
    model = model_name or f"gemini:{extract.GEMINI_MODEL}"
    app = FastAPI(title="Medical Bill Review", docs_url="/api/docs", redoc_url=None)
    app.state.store = store

    def run_job(doc_id: str) -> None:
        path = store.upload_path(doc_id)
        try:
            processed = processor(path)
            # uploads are stored under a random id; report the name the user knows
            processed.bill.source_file = store.get_document(doc_id)["filename"]
            store.save_result(doc_id, processed, model)
        except BillOCRError as e:
            logger.error("Processing %s failed: %s", doc_id, e)
            store.mark_failed(doc_id, str(e))
        except Exception as e:  # noqa: BLE001 -- a crash must not leave it "processing"
            logger.exception("Unexpected failure processing %s", doc_id)
            store.mark_failed(doc_id, f"unexpected: {e}")

    def get_or_404(doc_id: str) -> dict:
        doc = store.get_document(doc_id)
        if doc is None:
            raise HTTPException(404, "Document not found")
        return doc

    def with_review_flag(doc: dict) -> dict:
        if doc.get("current"):
            doc["needs_review"] = needs_human_review(BillData.from_dict(doc["current"]))
        return doc

    # ── pages ───────────────────────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    # ── documents ───────────────────────────────────────────────────────

    @app.post("/api/documents", status_code=201)
    async def upload(background: BackgroundTasks, files: list[UploadFile] = File(...)):
        created = []
        for f in files:
            name = Path(f.filename or "upload").name
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTS:
                raise HTTPException(400, f"Unsupported file type '{ext}' for {name}. "
                                         f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}")
            content = await f.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{name} is larger than 25 MB")
            if not content:
                raise HTTPException(400, f"{name} is empty")
            doc_id = store.create_document(name, content)
            background.add_task(run_job, doc_id)
            created.append({"id": doc_id, "filename": name, "status": STATUS_PROCESSING})
        return {"documents": created}

    @app.get("/api/documents")
    def list_documents():
        docs = store.list_documents()
        for d in docs:
            d["needs_review"] = needs_human_review(
                BillData(validation_flags=d["validation_flags"]))
        return {"documents": docs}

    @app.get("/api/documents/{doc_id}")
    def get_document(doc_id: str):
        return with_review_flag(get_or_404(doc_id))

    @app.put("/api/documents/{doc_id}")
    def save_document(doc_id: str, payload: dict = Body(...)):
        """Save the reviewer's version. Validation re-runs on the corrected data."""
        doc = get_or_404(doc_id)
        if doc["status"] not in (STATUS_READY, STATUS_REVIEWED):
            raise HTTPException(409, f"Document is {doc['status']}, nothing to review")
        bill = validate(_clean_bill(payload, doc["current"]))
        store.save_review(doc_id, bill)
        return with_review_flag(get_or_404(doc_id))

    @app.delete("/api/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str):
        if not store.delete(doc_id):
            raise HTTPException(404, "Document not found")

    @app.post("/api/documents/{doc_id}/reprocess", status_code=202)
    def reprocess(doc_id: str, background: BackgroundTasks):
        doc = get_or_404(doc_id)
        if doc["status"] not in (STATUS_FAILED, STATUS_READY):
            raise HTTPException(409, f"Cannot reprocess a {doc['status']} document")
        store.mark_processing(doc_id)
        background.add_task(run_job, doc_id)
        return {"id": doc_id, "status": STATUS_PROCESSING}

    @app.get("/api/documents/{doc_id}/pages/{n}")
    def page_image(doc_id: str, n: int):
        path = store.page_path(doc_id, n)
        if n < 0 or not path.exists():
            raise HTTPException(404, "Page not found")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/documents/{doc_id}/original")
    def original(doc_id: str):
        doc = get_or_404(doc_id)
        return FileResponse(store.upload_path(doc_id), filename=doc["filename"])

    @app.get("/api/documents/{doc_id}/locate")
    def locate_value(doc_id: str, value: str):
        """Boxes (fractions of page size) where `value` appears on the pages."""
        doc = get_or_404(doc_id)
        return {"boxes": locate(value, [p["words"] for p in doc["pages"]])}

    @app.get("/api/documents/{doc_id}/revisions")
    def revisions(doc_id: str):
        get_or_404(doc_id)
        return {"revisions": store.revisions(doc_id)}

    # ── accuracy + export ───────────────────────────────────────────────

    @app.get("/api/accuracy")
    def accuracy():
        return accuracy_report(store.review_records())

    @app.get("/api/export")
    def export(reviewed_only: bool = True):
        bills = store.bills(reviewed_only=reviewed_only)
        if not bills:
            raise HTTPException(404, "No reviewed bills to export yet" if reviewed_only
                                else "No processed bills to export yet")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = store.data_dir / "exports" / f"bills_{stamp}.xlsx"
        export_excel(bills, out, also_csv=False)
        return FileResponse(out, filename=out.name, media_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))

    return app


def _clean_bill(payload: dict, existing: dict | None) -> BillData:
    """
    Turn the browser's JSON into a BillData. Provenance fields (source file,
    OCR confidence, extraction method) always come from the stored record,
    never from the client.
    """
    def text(v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    def number(v, field_name):
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        try:
            return float(str(v).replace("$", "").replace(",", "").strip())
        except ValueError:
            raise HTTPException(422, f"'{v}' is not a number ({field_name})")

    data = {}
    for f in HEADER_FIELDS:
        data[f] = number(payload.get(f), f) if f in MONEY_FIELDS else text(payload.get(f))

    items = []
    for i, raw in enumerate(payload.get("line_items") or []):
        if not isinstance(raw, dict):
            continue
        item = {
            "description": text(raw.get("description")),
            "date": text(raw.get("date")),
            "quantity": number(raw.get("quantity"), f"line {i + 1} quantity"),
            "amount": number(raw.get("amount"), f"line {i + 1} amount"),
        }
        if all(item[f] is None for f in ITEM_FIELDS):
            continue  # fully blank row
        item["description"] = item["description"] or ""
        items.append(LineItem(**item))

    existing = existing or {}
    return BillData(
        line_items=items,
        source_file=existing.get("source_file"),
        ocr_mean_confidence=existing.get("ocr_mean_confidence"),
        extraction_method=existing.get("extraction_method"),
        **data,
    )
