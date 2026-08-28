"""Temporary: full pipeline test -- OCR one bill image, extract fields, show JSON."""

import sys
import json
import logging
import pathlib

sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from bill_ocr.ocr import ocr_image_file
from bill_ocr.extract import extract_fields

image = sorted(pathlib.Path("data/input").glob("*.png"))[0]
print(f"=== {image.name} ===")

ocr_result = ocr_image_file(image)
print(f"OCR: {ocr_result.word_count} words, mean conf {ocr_result.mean_confidence:.1f}")

bill = extract_fields(
    ocr_result.text,
    source_file=image.name,
    ocr_confidence=ocr_result.mean_confidence,
)

print(json.dumps(bill.to_dict(), indent=2))