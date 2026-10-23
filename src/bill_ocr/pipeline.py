"""
Pipeline orchestrator: folder of mixed bills -> one reviewed Excel workbook.

Per file: classify -> route (digital text / render+OCR / preprocess+OCR)
          -> Gemini extract -> local validate -> collect.
Failures are isolated per file: one bad bill never kills the batch.

process_file_detailed() additionally returns each page as an image plus its
word boxes (normalized 0..1 coordinates), which the review UI uses to show
where on the page each extracted value came from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from bill_ocr.errors import BillOCRError, FileReadError
from bill_ocr.export import export_excel
from bill_ocr.extract import extract_fields
from bill_ocr.ingest import classify, SourceType
from bill_ocr.ocr import OCRResult, ocr_array
from bill_ocr.preprocess import preprocess_array
from bill_ocr.schema import BillData
from bill_ocr.validate import validate

logger = logging.getLogger(__name__)

RENDER_DPI = 300   # print-quality rasterization for scanned PDFs (OCR input)
DISPLAY_DPI = 150  # digital PDFs are only rendered for the review UI


@dataclass
class PageWord:
    """One word on a page. Coordinates are fractions of page width/height."""
    text: str
    confidence: float | None   # None for digital text (exact, no OCR)
    x: float
    y: float
    w: float
    h: float


@dataclass
class PageView:
    image: np.ndarray          # BGR image of the page as the reviewer should see it
    words: list[PageWord] = field(default_factory=list)


@dataclass
class ProcessedDocument:
    bill: BillData
    source_type: SourceType
    pages: list[PageView] = field(default_factory=list)


def process_file(path: str | Path, debug_dir: str | Path | None = None) -> BillData:
    """Run one file through the full pipeline. Raises BillOCRError subtypes on failure."""
    return process_file_detailed(path, debug_dir=debug_dir, with_pages=False).bill


def process_file_detailed(path: str | Path, debug_dir: str | Path | None = None,
                          with_pages: bool = True) -> ProcessedDocument:
    """
    Full pipeline for one file, keeping page images and word boxes.
    with_pages=False skips rendering digital PDFs (the CLI does not need images).
    """
    path = Path(path)
    doc = classify(path)

    if doc.source_type == SourceType.DIGITAL_PDF:
        text, pages = _read_digital_pdf(path, with_pages)
        confidence = None
    elif doc.source_type == SourceType.SCANNED_PDF:
        text, confidence, pages = _ocr_scanned_pdf(path, debug_dir)
    else:  # IMAGE
        text, confidence, pages = _ocr_image(path, debug_dir)

    bill = extract_fields(text, source_file=path.name, ocr_confidence=confidence)
    return ProcessedDocument(
        bill=validate(bill),
        source_type=doc.source_type,
        pages=pages if with_pages else [],
    )


def process_directory(input_dir: str | Path, output_path: str | Path,
                      debug_dir: str | Path | None = None) -> tuple[list[BillData], list[str]]:
    """
    Process every supported file in a folder; export one workbook.
    Returns (successful bills, failure messages) -- failures never abort the batch.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise BillOCRError(f"Input folder not found: {input_dir}")

    files = sorted(p for p in input_dir.iterdir() if p.is_file())
    if not files:
        raise BillOCRError(f"No files found in {input_dir}")

    bills: list[BillData] = []
    failures: list[str] = []

    for path in files:
        logger.info("── Processing %s", path.name)
        try:
            bills.append(process_file(path, debug_dir=debug_dir))
        except BillOCRError as e:
            logger.error("FAILED %s: %s", path.name, e)
            failures.append(f"{path.name}: {e}")
        except Exception as e:  # noqa: BLE001 -- unknown bugs also must not kill the batch
            logger.exception("UNEXPECTED failure on %s", path.name)
            failures.append(f"{path.name}: unexpected: {e}")

    if bills:
        export_excel(bills, output_path)
    logger.info("Batch done: %d succeeded, %d failed", len(bills), len(failures))
    return bills, failures


def _read_digital_pdf(path: Path, with_pages: bool) -> tuple[str, list[PageView]]:
    """Digital PDFs: read the embedded text layer directly -- exact, instant, no OCR."""
    doc = pymupdf.open(path)
    try:
        texts, pages = [], []
        for page in doc:
            texts.append(page.get_text())
            if not with_pages:
                continue
            pw, ph = page.rect.width, page.rect.height
            words = [
                PageWord(text=w[4], confidence=None,
                         x=w[0] / pw, y=w[1] / ph, w=(w[2] - w[0]) / pw, h=(w[3] - w[1]) / ph)
                for w in page.get_text("words")
            ]
            pages.append(PageView(image=_render(page, DISPLAY_DPI), words=words))
    finally:
        doc.close()
    return "\n\n".join(texts), pages


def _ocr_scanned_pdf(path: Path, debug_dir: str | Path | None
                     ) -> tuple[str, float, list[PageView]]:
    """Scanned PDFs: rasterize each page at 300 DPI in memory, preprocess + OCR, merge."""
    doc = pymupdf.open(path)
    try:
        texts, confidences, pages = [], [], []
        for i, page in enumerate(doc):
            img = _render(page, RENDER_DPI)
            binary = preprocess_array(img, name=f"{path.stem}_p{i+1}", debug_dir=debug_dir)
            result = ocr_array(binary)
            texts.append(result.text)
            confidences.append(result.mean_confidence)
            pages.append(PageView(image=img, words=_normalize_words(result, binary.shape)))
    finally:
        doc.close()

    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return "\n\n".join(texts), mean_conf, pages


def _ocr_image(path: Path, debug_dir: str | Path | None) -> tuple[str, float, list[PageView]]:
    img = cv2.imread(str(path))
    if img is None:
        raise FileReadError(f"Could not read image: {path}")
    binary = preprocess_array(img, name=path.stem, debug_dir=debug_dir)
    result = ocr_array(binary)
    page = PageView(image=img, words=_normalize_words(result, binary.shape))
    return result.text, result.mean_confidence, [page]


def _render(page: pymupdf.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _normalize_words(result: OCRResult, shape: tuple[int, ...]) -> list[PageWord]:
    """
    OCR boxes are in preprocessed-image pixels (possibly upscaled 2x). Dividing by
    that image's size gives fractions valid on the original page too. Deskew
    rotation (at most a few degrees) is not undone, so boxes can drift slightly.
    """
    h, w = shape[:2]
    return [
        PageWord(text=word.text, confidence=word.confidence,
                 x=word.left / w, y=word.top / h, w=word.width / w, h=word.height / h)
        for word in result.words
    ]
