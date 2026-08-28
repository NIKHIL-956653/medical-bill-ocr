"""Temporary: OCR every image in data/input and show text + confidence."""

import sys
import logging
import pathlib

sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from bill_ocr.ocr import ocr_image_file

for p in sorted(pathlib.Path("data/input").glob("*.png")):
    print("=" * 70)
    print(f"FILE: {p.name}")
    print("=" * 70)
    result = ocr_image_file(p, debug_dir="artifacts/preprocessed")
    print(f"words: {result.word_count} | mean conf: {result.mean_confidence:.1f} | low-conf words: {result.low_conf_count}")
    print("-" * 70)
    print(result.text[:1500])
    print()