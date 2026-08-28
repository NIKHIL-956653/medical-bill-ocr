"""
Ingestion router: classify each input file into a processing route.

Routes:
    DIGITAL_PDF -- PDF with a real text layer -> extract text directly (no OCR)
    SCANNED_PDF -- PDF that is just wrapped images -> render pages -> OCR
    IMAGE       -- photo/scan file (jpg/png/...) -> preprocess -> OCR

The router exists because OCR is the slowest, least accurate step --
we only pay that cost when a file genuinely has no text layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pymupdf

from bill_ocr.errors import FileReadError, UnsupportedFileError

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}

# A digital PDF page usually carries hundreds/thousands of chars.
# A scanned one yields ~0 (occasionally a few stray chars of metadata).
# 50 is a safe boundary between those two worlds.
TEXT_LAYER_THRESHOLD = 50


class SourceType(str, Enum):
    DIGITAL_PDF = "digital_pdf"
    SCANNED_PDF = "scanned_pdf"
    IMAGE = "image"


@dataclass(frozen=True)
class InputDocument:
    """Everything the rest of the pipeline needs to know about one input file."""

    path: Path
    source_type: SourceType
    n_pages: int          # images count as 1
    text_chars: int       # chars found in PDF text layer (0 for images)


def classify(path: str | Path) -> InputDocument:
    """
    Classify one input file into its processing route.

    Raises:
        UnsupportedFileError: extension we don't handle.
        FileReadError: file missing, empty, or unreadable as its claimed type.
    """
    path = Path(path)

    if not path.exists():
        raise FileReadError(f"File not found: {path}")
    if path.stat().st_size == 0:
        raise FileReadError(f"File is empty (0 bytes): {path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        return _classify_pdf(path)
    if ext in SUPPORTED_IMAGE_EXTS:
        logger.info("%s -> IMAGE route", path.name)
        return InputDocument(path=path, source_type=SourceType.IMAGE, n_pages=1, text_chars=0)

    raise UnsupportedFileError(
        f"Unsupported file type '{ext}' for {path.name}. "
        f"Supported: .pdf, {', '.join(sorted(SUPPORTED_IMAGE_EXTS))}"
    )


def _classify_pdf(path: Path) -> InputDocument:
    try:
        doc = pymupdf.open(path)
    except Exception as e:
        raise FileReadError(f"Could not open PDF {path.name}: {e}") from e

    try:
        n_pages = len(doc)
        text_chars = sum(len(page.get_text()) for page in doc)
    finally:
        doc.close()

    if n_pages == 0:
        raise FileReadError(f"PDF has no pages: {path.name}")

    if text_chars >= TEXT_LAYER_THRESHOLD:
        source_type = SourceType.DIGITAL_PDF
        logger.info("%s -> DIGITAL_PDF route (%d chars)", path.name, text_chars)
    else:
        source_type = SourceType.SCANNED_PDF
        logger.info("%s -> SCANNED_PDF route (only %d chars)", path.name, text_chars)

    return InputDocument(path=path, source_type=source_type, n_pages=n_pages, text_chars=text_chars)