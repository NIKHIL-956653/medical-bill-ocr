"""
Pipeline orchestrator: folder of mixed bills -> one reviewed Excel workbook.

Per file: classify -> route (digital text / render+OCR / preprocess+OCR)
          -> Gemini extract -> local validate -> collect.
Failures are isolated per file: one bad bill never kills the batch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

from bill_ocr.errors import BillOCRError
from bill_ocr.export import export_excel
from bill_ocr.extract import extract_fields
from bill_ocr.ingest import classify, SourceType
from bill_ocr.ocr import ocr_image_file, ocr_array
from bill_ocr.preprocess import preprocess_image
from bill_ocr.schema import BillData
from bill_ocr.validate import validate

logger = logging.getLogger(__name__)

RENDER_DPI = 300  # print-quality rasterization for scanned PDFs


def process_file(path: str | Path, debug_dir: str | Path | None = None) -> BillData:
    """Run one file through the full pipeline. Raises BillOCRError subtypes on failure."""
    path = Path(path)
    doc = classify(path)

    if doc.source_type == SourceType.DIGITAL_PDF:
        text, confidence = _read_digital_pdf(path), None
    elif doc.source_type == SourceType.SCANNED_PDF:
        text, confidence = _ocr_scanned_pdf(path, debug_dir)
    else:  # IMAGE
        result = ocr_image_file(path, debug_dir=debug_dir)
        text, confidence = result.text, result.mean_confidence

    bill = extract_fields(text, source_file=path.name, ocr_confidence=confidence)
    return validate(bill)


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


def _read_digital_pdf(path: Path) -> str:
    """Digital PDFs: read the embedded text layer directly -- exact, instant, no OCR."""
    doc = pymupdf.open(path)
    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()
    return "\n\n".join(pages)


def _ocr_scanned_pdf(path: Path, debug_dir: str | Path | None) -> tuple[str, float]:
    """Scanned PDFs: rasterize each page at 300 DPI, preprocess + OCR, merge."""
    import cv2
    import numpy as np

    doc = pymupdf.open(path)
    try:
        texts, confidences = [], []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=RENDER_DPI)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n >= 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # reuse the image path: write rendered page, preprocess, OCR
            tmp = Path(debug_dir or "artifacts") / "rendered" / f"{path.stem}_p{i+1}.png"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tmp), img)

            binary = preprocess_image(tmp, debug_dir=debug_dir)
            result = ocr_array(binary)
            texts.append(result.text)
            confidences.append(result.mean_confidence)
    finally:
        doc.close()

    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return "\n\n".join(texts), mean_conf