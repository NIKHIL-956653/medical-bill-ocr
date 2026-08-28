"""
Validation layer: deterministic local checks on extracted BillData.

The LLM's output is untrusted input. This module does the boring math
that decides whether we believe it -- and flags anything unverifiable
for human review instead of silently passing it through.
"""

from __future__ import annotations

import logging
from datetime import datetime

from bill_ocr.schema import BillData

logger = logging.getLogger(__name__)

# tolerance for float comparison on money (covers rounding artifacts)
MONEY_TOLERANCE = 0.05
# OCR mean confidence below this contributes a review flag
LOW_OCR_CONFIDENCE = 65.0


def validate(bill: BillData) -> BillData:
    """
    Run all checks; append human-readable flags to bill.validation_flags.
    Returns the same BillData (mutated) for easy chaining.
    """
    flags: list[str] = []

    flags += _check_line_item_sum(bill)
    flags += _check_amount_sanity(bill)
    flags += _check_required_fields(bill)
    flags += _check_ocr_confidence(bill)

    bill.validation_flags = flags
    if flags:
        logger.warning("%s: %d validation flag(s): %s",
                       bill.source_file, len(flags), "; ".join(flags))
    else:
        logger.info("%s: all validation checks passed", bill.source_file)
    return bill


def _check_line_item_sum(bill: BillData) -> list[str]:
    """Do the line items reconcile with the printed subtotal/total?"""
    amounts = [li.amount for li in bill.line_items if li.amount is not None]
    if not amounts:
        return ["no_line_item_amounts"]

    items_sum = round(sum(amounts), 2)
    flags = []

    # Compare against whichever printed figure exists (subtotal preferred)
    reference = bill.subtotal if bill.subtotal is not None else bill.total
    if reference is not None:
        if abs(items_sum - reference) <= MONEY_TOLERANCE:
            logger.info("Sum check OK: items %.2f == printed %.2f", items_sum, reference)
        else:
            flags.append(
                f"totals_mismatch(items={items_sum:.2f}, printed={reference:.2f})"
            )
    else:
        flags.append("no_printed_total_to_verify")

    return flags


def _check_amount_sanity(bill: BillData) -> list[str]:
    flags = []
    if bill.total is not None and bill.total < 0:
        flags.append("negative_total")
    for li in bill.line_items:
        if li.quantity is not None and abs(li.quantity) > 1000:
            flags.append(f"suspicious_quantity({li.description[:30]}={li.quantity})")
    return flags


def _check_required_fields(bill: BillData) -> list[str]:
    flags = []
    if not bill.provider_name:
        flags.append("missing_provider_name")
    if not bill.invoice_number:
        flags.append("missing_invoice_number")
    if not bill.bill_date:
        flags.append("missing_bill_date")
    elif not _parseable_date(bill.bill_date):
        flags.append(f"unparseable_date({bill.bill_date})")
    return flags


def _check_ocr_confidence(bill: BillData) -> list[str]:
    if bill.ocr_mean_confidence is not None and bill.ocr_mean_confidence < LOW_OCR_CONFIDENCE:
        return [f"low_ocr_confidence({bill.ocr_mean_confidence:.1f})"]
    return []


def _parseable_date(text: str) -> bool:
    """Try the formats bills actually use."""
    formats = ["%m/%d/%Y", "%m-%d-%y", "%m-%d-%Y", "%d/%m/%Y",
               "%Y-%m-%d", "%b %d, %Y", "%d %b %Y", "%m/%d/%y"]
    for fmt in formats:
        try:
            datetime.strptime(text.strip(), fmt)
            return True
        except ValueError:
            continue
    return False


def needs_human_review(bill: BillData) -> bool:
    """
    The headline decision: can this row go straight to Excel, or does a
    human need to look? Any totals mismatch, or low confidence combined
    with any other flag, escalates.
    """
    flags = bill.validation_flags
    if any(f.startswith("totals_mismatch") for f in flags):
        return True
    has_low_conf = any(f.startswith("low_ocr_confidence") for f in flags)
    return has_low_conf and len(flags) > 1