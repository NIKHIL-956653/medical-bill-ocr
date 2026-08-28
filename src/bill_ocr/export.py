"""
Export layer: list[BillData] -> Excel workbook (+ optional CSV pair).

Sheet 1 "Bills":      one row per bill, review status first.
Sheet 2 "Line Items": one row per line item, joined by invoice number.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from bill_ocr.errors import BillOCRError
from bill_ocr.schema import BillData
from bill_ocr.validate import needs_human_review

logger = logging.getLogger(__name__)


class ExportError(BillOCRError):
    """Raised when writing output files fails."""


def bills_to_frames(bills: list[BillData]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the two DataFrames (bills, line items) from extracted data."""
    bill_rows = []
    item_rows = []

    for bill in bills:
        review = needs_human_review(bill)
        bill_rows.append({
            "NEEDS_REVIEW": "YES" if review else "",
            "validation_flags": "; ".join(bill.validation_flags),
            "source_file": bill.source_file,
            "provider_name": bill.provider_name,
            "invoice_number": bill.invoice_number,
            "bill_date": bill.bill_date,
            "client_name": bill.client_name,
            "client_id": bill.client_id,
            "line_item_count": len(bill.line_items),
            "subtotal": bill.subtotal,
            "payments": bill.payments,
            "balance_due": bill.balance_due,
            "total": bill.total,
            "ocr_confidence": bill.ocr_mean_confidence,
            "extraction_method": bill.extraction_method,
        })

        for li in bill.line_items:
            item_rows.append({
                "invoice_number": bill.invoice_number,
                "source_file": bill.source_file,
                "date": li.date,
                "description": li.description,
                "quantity": li.quantity,
                "amount": li.amount,
            })

    return pd.DataFrame(bill_rows), pd.DataFrame(item_rows)


def export_excel(bills: list[BillData], output_path: str | Path,
                 also_csv: bool = True) -> Path:
    """
    Write the workbook (and optional CSV pair alongside it).

    Raises:
        ExportError: nothing to export, or file write failure
        (e.g. the workbook is open in Excel -- Windows locks it).
    """
    if not bills:
        raise ExportError("No bills to export.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bills_df, items_df = bills_to_frames(bills)

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            bills_df.to_excel(writer, sheet_name="Bills", index=False)
            items_df.to_excel(writer, sheet_name="Line Items", index=False)
            _autofit_columns(writer, "Bills", bills_df)
            _autofit_columns(writer, "Line Items", items_df)
    except PermissionError as e:
        raise ExportError(
            f"Cannot write {output_path.name} -- is it open in Excel? Close it and retry."
        ) from e

    logger.info("Wrote %s (%d bills, %d line items)",
                output_path, len(bills_df), len(items_df))

    if also_csv:
        bills_csv = output_path.with_name(output_path.stem + "_bills.csv")
        items_csv = output_path.with_name(output_path.stem + "_line_items.csv")
        bills_df.to_csv(bills_csv, index=False)
        items_df.to_csv(items_csv, index=False)
        logger.info("Wrote CSV pair: %s, %s", bills_csv.name, items_csv.name)

    return output_path


def _autofit_columns(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    """Reasonable column widths so the sheet is readable on open."""
    worksheet = writer.sheets[sheet_name]
    for i, column in enumerate(df.columns, start=1):
        max_len = max(
            len(str(column)),
            df[column].astype(str).str.len().max() if len(df) else 0,
        )
        worksheet.column_dimensions[
            worksheet.cell(row=1, column=i).column_letter
        ].width = min(max_len + 3, 50)