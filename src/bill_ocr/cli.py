"""
CLI for the bill OCR pipeline.

Usage:
    python -m bill_ocr process --input data/input --output data/output/results.xlsx
    python -m bill_ocr process --input data/input --output out.xlsx --debug-dir artifacts/debug -v
"""

from __future__ import annotations

import argparse
import logging
import sys

from bill_ocr.errors import BillOCRError
from bill_ocr.pipeline import process_directory

logger = logging.getLogger("bill_ocr.cli")


def cmd_process(args: argparse.Namespace) -> int:
    bills, failures = process_directory(
        args.input, args.output, debug_dir=args.debug_dir
    )

    flagged = sum(1 for b in bills if b.validation_flags)
    print()
    print(f"Processed : {len(bills) + len(failures)} file(s)")
    print(f"Succeeded : {len(bills)}  ({flagged} flagged for review)")
    print(f"Failed    : {len(failures)}")
    for f in failures:
        print(f"  - {f}")
    if bills:
        print(f"Output    : {args.output}")
    return 0 if not failures else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bill_ocr",
        description="Medical Bill OCR: scans/PDFs -> validated Excel.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="Process a folder of bills into Excel.")
    p.add_argument("--input", required=True, help="Folder containing bill files.")
    p.add_argument("--output", required=True, help="Path of the .xlsx to write.")
    p.add_argument("--debug-dir", default=None,
                   help="Optional: write preprocessing stage images here.")
    p.set_defaults(func=cmd_process)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        return args.func(args)
    except BillOCRError as e:
        logger.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())