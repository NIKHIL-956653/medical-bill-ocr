"""
Local storage for the review UI: uploaded bills, extraction results,
page images, and every human correction.

Layout under data_dir (default data/review -- git-ignored like all of data/):
    review.db          SQLite: documents + revisions
    uploads/<id><ext>  original files, stored under a random id (never the client's name)
    pages/<id>_<n>.jpg page images shown in the UI

Each document keeps two snapshots: `extracted` (pipeline output, never modified)
and `current` (latest reviewed version). Every save also appends a full
snapshot to `revisions`, so the correction history is auditable.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import cv2

from bill_ocr.accuracy import ReviewRecord
from bill_ocr.schema import BillData

MAX_PAGE_WIDTH = 1700
JPEG_QUALITY = 85

STATUS_PROCESSING = "processing"
STATUS_READY = "ready"          # extracted, waiting for a human
STATUS_REVIEWED = "reviewed"    # a human saved it
STATUS_FAILED = "failed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id             TEXT PRIMARY KEY,
    filename       TEXT NOT NULL,
    stored_path    TEXT NOT NULL,
    status         TEXT NOT NULL,
    error          TEXT,
    source_type    TEXT,
    model          TEXT,
    extracted_json TEXT,
    current_json   TEXT,
    pages_json     TEXT,
    created_at     TEXT NOT NULL,
    reviewed_at    TEXT
);
CREATE TABLE IF NOT EXISTS revisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    saved_at    TEXT NOT NULL,
    bill_json   TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReviewStore:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.uploads_dir = self.data_dir / "uploads"
        self.pages_dir = self.data_dir / "pages"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "review.db"
        with closing(self._connect()) as db, db:
            db.executescript(_SCHEMA)
            # a restart kills in-flight background jobs; don't leave them spinning forever
            db.execute("UPDATE documents SET status=?, error=? WHERE status=?",
                       (STATUS_FAILED, "interrupted by server restart", STATUS_PROCESSING))

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    # ── documents ────────────────────────────────────────────────────────

    def create_document(self, filename: str, content: bytes) -> str:
        doc_id = uuid.uuid4().hex
        ext = Path(filename).suffix.lower()
        stored = self.uploads_dir / f"{doc_id}{ext}"
        stored.write_bytes(content)
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO documents (id, filename, stored_path, status, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, Path(filename).name, str(stored), STATUS_PROCESSING, _now()))
        return doc_id

    def save_result(self, doc_id: str, processed, model: str | None) -> None:
        """Store a pipeline.ProcessedDocument: page images, word boxes, extraction."""
        pages = []
        for n, page in enumerate(processed.pages):
            img = page.image
            if img.shape[1] > MAX_PAGE_WIDTH:
                scale = MAX_PAGE_WIDTH / img.shape[1]
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(self.page_path(doc_id, n)), img,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            pages.append({
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
                "words": [[w.text, w.confidence, round(w.x, 5), round(w.y, 5),
                           round(w.w, 5), round(w.h, 5)] for w in page.words],
            })
        bill_json = json.dumps(processed.bill.to_dict())
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE documents SET status=?, error=NULL, source_type=?, model=?, "
                "extracted_json=?, current_json=?, pages_json=? WHERE id=?",
                (STATUS_READY, processed.source_type.value, model,
                 bill_json, bill_json, json.dumps(pages), doc_id))

    def mark_processing(self, doc_id: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute("UPDATE documents SET status=?, error=NULL WHERE id=?",
                       (STATUS_PROCESSING, doc_id))

    def mark_failed(self, doc_id: str, error: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute("UPDATE documents SET status=?, error=? WHERE id=?",
                       (STATUS_FAILED, error, doc_id))

    def save_review(self, doc_id: str, bill: BillData) -> None:
        bill_json = json.dumps(bill.to_dict())
        now = _now()
        with closing(self._connect()) as db, db:
            db.execute("UPDATE documents SET status=?, current_json=?, reviewed_at=? WHERE id=?",
                       (STATUS_REVIEWED, bill_json, now, doc_id))
            db.execute("INSERT INTO revisions (document_id, saved_at, bill_json) VALUES (?, ?, ?)",
                       (doc_id, now, bill_json))

    def delete(self, doc_id: str) -> bool:
        row = self._row(doc_id)
        if row is None:
            return False
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        Path(row["stored_path"]).unlink(missing_ok=True)
        for p in self.pages_dir.glob(f"{doc_id}_*.jpg"):
            p.unlink(missing_ok=True)
        return True

    # ── reads ────────────────────────────────────────────────────────────

    def page_path(self, doc_id: str, n: int) -> Path:
        return self.pages_dir / f"{doc_id}_{n}.jpg"

    def upload_path(self, doc_id: str) -> Path | None:
        row = self._row(doc_id)
        return Path(row["stored_path"]) if row else None

    def _row(self, doc_id: str) -> sqlite3.Row | None:
        with closing(self._connect()) as db:
            return db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()

    def list_documents(self) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT id, filename, status, error, current_json, created_at, "
                              "reviewed_at FROM documents ORDER BY created_at DESC, rowid DESC"
                              ).fetchall()
        out = []
        for r in rows:
            current = json.loads(r["current_json"]) if r["current_json"] else {}
            out.append({
                "id": r["id"], "filename": r["filename"], "status": r["status"],
                "error": r["error"], "created_at": r["created_at"],
                "reviewed_at": r["reviewed_at"],
                "provider_name": current.get("provider_name"),
                "total": current.get("total"),
                "validation_flags": current.get("validation_flags") or [],
            })
        return out

    def get_document(self, doc_id: str) -> dict | None:
        r = self._row(doc_id)
        if r is None:
            return None
        return {
            "id": r["id"], "filename": r["filename"], "status": r["status"],
            "error": r["error"], "source_type": r["source_type"], "model": r["model"],
            "created_at": r["created_at"], "reviewed_at": r["reviewed_at"],
            "extracted": json.loads(r["extracted_json"]) if r["extracted_json"] else None,
            "current": json.loads(r["current_json"]) if r["current_json"] else None,
            "pages": json.loads(r["pages_json"]) if r["pages_json"] else [],
        }

    def revisions(self, doc_id: str) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT saved_at, bill_json FROM revisions WHERE document_id=? "
                              "ORDER BY id", (doc_id,)).fetchall()
        return [{"saved_at": r["saved_at"], "bill": json.loads(r["bill_json"])} for r in rows]

    def review_records(self) -> list[ReviewRecord]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT extracted_json, current_json, reviewed_at, model "
                              "FROM documents WHERE status=?", (STATUS_REVIEWED,)).fetchall()
        return [ReviewRecord(json.loads(r["extracted_json"]), json.loads(r["current_json"]),
                             r["reviewed_at"], r["model"]) for r in rows]

    def bills(self, reviewed_only: bool) -> list[BillData]:
        statuses = (STATUS_REVIEWED,) if reviewed_only else (STATUS_REVIEWED, STATUS_READY)
        marks = ",".join("?" * len(statuses))
        with closing(self._connect()) as db:
            rows = db.execute(f"SELECT current_json FROM documents WHERE status IN ({marks}) "
                              "ORDER BY created_at, rowid", statuses).fetchall()
        return [BillData.from_dict(json.loads(r["current_json"])) for r in rows]
