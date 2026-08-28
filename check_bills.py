"""
Inventory of bill files in data/input/:
- file type and size
- for PDFs: page count + does it have an extractable text layer (digital) or not (scanned)?
- for images: dimensions (resolution matters hugely for OCR quality)

Temporary scaffold -- this logic graduates into the real pipeline next.
"""

from pathlib import Path

import fitz  # pymupdf
from PIL import Image

INPUT_DIR = Path("data/input")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}


def inspect_pdf(path: Path) -> str:
    doc = fitz.open(path)
    n_pages = len(doc)
    total_chars = sum(len(page.get_text()) for page in doc)
    doc.close()
    kind = "DIGITAL (has text layer)" if total_chars > 50 else "SCANNED (no text layer -> OCR)"
    return f"PDF  {n_pages:2d} page(s), {total_chars:6d} chars -> {kind}"


def inspect_image(path: Path) -> str:
    with Image.open(path) as img:
        w, h = img.size
    quality = "good" if min(w, h) >= 1000 else "LOW-RES (OCR may struggle)"
    return f"IMG  {w}x{h} px -> {quality}"


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"Missing folder: {INPUT_DIR}")
        return

    files = sorted(p for p in INPUT_DIR.iterdir() if p.is_file())
    if not files:
        print(f"No files found in {INPUT_DIR} -- copy the bills in first.")
        return

    print(f"Found {len(files)} file(s) in {INPUT_DIR}:\n")
    for p in files:
        size_kb = p.stat().st_size / 1024
        try:
            if p.suffix.lower() == ".pdf":
                info = inspect_pdf(p)
            elif p.suffix.lower() in IMAGE_EXTS:
                info = inspect_image(p)
            else:
                info = "UNSUPPORTED type (will be skipped by pipeline)"
        except Exception as e:  # noqa: BLE001 -- inventory must never crash on one bad file
            info = f"ERROR reading file: {e}"
        print(f"  {p.name:40s} {size_kb:8.1f} KB  {info}")


if __name__ == "__main__":
    main()