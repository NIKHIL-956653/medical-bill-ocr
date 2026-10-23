"""
CLI for the bill OCR pipeline.

Usage:
    python -m bill_ocr process --input data/input --output data/output/results.xlsx
    python -m bill_ocr process --input data/input --output out.xlsx --debug-dir artifacts/debug -v
    python -m bill_ocr serve                  # review web UI on http://127.0.0.1:8000
    python -m bill_ocr accuracy               # extraction accuracy from reviewed bills
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


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
        from bill_ocr.web import create_app
    except ImportError as e:
        raise BillOCRError(
            f"Web UI dependencies missing ({e.name}). Run: pip install -e \".[web]\""
        ) from e
    if args.host not in ("127.0.0.1", "localhost"):
        logger.warning("Serving on %s: the review UI has no login and shows patient data. "
                       "Only do this on a trusted network.", args.host)
    print(f"Review UI: http://{args.host}:{args.port}  (data in {args.data_dir})")
    uvicorn.run(create_app(args.data_dir), host=args.host, port=args.port, log_level="info")
    return 0


def cmd_accuracy(args: argparse.Namespace) -> int:
    from bill_ocr.accuracy import accuracy_report
    from bill_ocr.store import ReviewStore

    report = accuracy_report(ReviewStore(args.data_dir).review_records())
    if not report["documents"]:
        print("No reviewed bills yet. Review some in the web UI first (python -m bill_ocr serve).")
        return 0

    pct = lambda v: "-" if v is None else f"{v * 100:.1f}%"  # noqa: E731
    print(f"Reviewed bills       : {report['documents']}")
    print(f"Fields correct       : {pct(report['field_accuracy'])}")
    print(f"Bills w/o corrections: {pct(report['perfect_rate'])}")
    print()
    print(f"{'field':<26}{'correct':>12}{'accuracy':>10}")
    for f in report["fields"]:
        print(f"{f['field']:<26}{str(f['correct']) + '/' + str(f['total']):>12}"
              f"{pct(f['accuracy']):>10}")
    for key, label in (("by_week", "week"), ("by_model", "model")):
        print()
        print(f"{label:<26}{'bills':>12}{'fields':>10}{'perfect':>10}")
        for row in report[key]:
            print(f"{row[label]:<26}{row['documents']:>12}{pct(row['field_accuracy']):>10}"
                  f"{pct(row['perfect_rate']):>10}")
    return 0


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

    s = sub.add_parser("serve", help="Start the review web UI.")
    s.add_argument("--data-dir", default="data/review",
                   help="Where uploads, page images and the review database live.")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    a = sub.add_parser("accuracy", help="Print extraction accuracy from reviewed bills.")
    a.add_argument("--data-dir", default="data/review")
    a.set_defaults(func=cmd_accuracy)

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