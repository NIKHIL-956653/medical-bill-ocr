"""Temporary: test Excel export with mock bills -- no API needed."""

import sys
import logging

sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from bill_ocr.schema import BillData, LineItem
from bill_ocr.validate import validate
from bill_ocr.export import export_excel

# Mock bill 1: clean -- sums reconcile
bill1 = BillData(
    provider_name="Carol Stream Animal Hospital",
    invoice_number="326099",
    bill_date="11/7/2024",
    client_name="Steven Gimre",
    line_items=[
        LineItem("Exam - Annual Physical", 1, 57.69),
        LineItem("Vaccine: Rabies Canine", 1, 28.87),
        LineItem("Heartworm Vaccine", 1, 91.12),
    ],
    subtotal=177.68,
    total=177.68,
    source_file="bill1_test.png",
    ocr_mean_confidence=78.0,
    extraction_method="gemini",
)

# Mock bill 2: broken -- sums DON'T reconcile + low confidence -> should flag
bill2 = BillData(
    provider_name="Ridgewood Animal Hospital",
    invoice_number="97986",
    bill_date="10-11-23",
    line_items=[
        LineItem("Litter exam", 1, 200.00),
        LineItem("DHPP vaccine", 7, 175.00),
    ],
    subtotal=589.00,          # items sum to 375 -> mismatch on purpose
    total=589.00,
    source_file="bill2_test.png",
    ocr_mean_confidence=58.0,  # below threshold on purpose
    extraction_method="gemini",
)

bills = [validate(bill1), validate(bill2)]
path = export_excel(bills, "data/output/results.xlsx")
print(f"\nDone -> {path}")
print("Open it and check: bill 2 should show NEEDS_REVIEW=YES with flags; bill 1 clean.")