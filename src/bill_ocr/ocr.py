"""
OCR wrapper around Tesseract (pytesseract).

Returns not just text but per-word confidence scores, plus a document-level
confidence summary -- so downstream stages can distinguish "trustworthy read"
from "the engine was guessing".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytesseract

from bill_ocr.errors import BillOCRError
from bill_ocr.preprocess import preprocess_image

logger = logging.getLogger(__name__)

# words below this confidence are counted as "low confidence" in the summary
LOW_CONF_THRESHOLD = 60


@dataclass
class OCRWord:
    text: str
    confidence: float
    # bounding box (pixels) -- lets us later locate a word on the page
    left: int
    top: int
    width: int
    height: int


@dataclass
class OCRResult:
    text: str                      # full recognized text, line breaks preserved
    words: list[OCRWord] = field(default_factory=list)
    mean_confidence: float = 0.0
    low_conf_count: int = 0

    @property
    def word_count(self) -> int:
        return len(self.words)


def ocr_array(binary_img: np.ndarray) -> OCRResult:
    """
    Run Tesseract on a preprocessed (binarized) image array.

    Raises:
        BillOCRError: if Tesseract itself fails to run.
    """
    try:
        data = pytesseract.image_to_data(
            binary_img,
            output_type=pytesseract.Output.DICT,
            config="--psm 6",  # assume a uniform block of text (good default for bills)
        )
    except pytesseract.TesseractError as e:
        raise BillOCRError(f"Tesseract failed: {e}") from e

    words: list[OCRWord] = []
    for i in range(len(data["text"])):
        word_text = data["text"][i].strip()
        conf = float(data["conf"][i])
        if not word_text or conf < 0:  # conf -1 marks non-word structural rows
            continue
        words.append(
            OCRWord(
                text=word_text,
                confidence=conf,
                left=data["left"][i],
                top=data["top"][i],
                width=data["width"][i],
                height=data["height"][i],
            )
        )

    full_text = pytesseract.image_to_string(binary_img, config="--psm 6")

    mean_conf = float(np.mean([w.confidence for w in words])) if words else 0.0
    low_conf = sum(1 for w in words if w.confidence < LOW_CONF_THRESHOLD)

    logger.info(
        "OCR: %d words, mean confidence %.1f, %d low-confidence",
        len(words), mean_conf, low_conf,
    )
    return OCRResult(
        text=full_text,
        words=words,
        mean_confidence=mean_conf,
        low_conf_count=low_conf,
    )


def ocr_image_file(path: str | Path, debug_dir: str | Path | None = None) -> OCRResult:
    """Convenience: preprocess an image file, then OCR it."""
    binary = preprocess_image(path, debug_dir=debug_dir)
    return ocr_array(binary)