"""
The data contract: what structured information we extract from every bill.

Every downstream module (validation, export) works against these classes,
never against raw text. Change the schema here, everything else follows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class LineItem:
    description: str
    quantity: float | None = None
    amount: float | None = None
    date: str | None = None          # as printed on the bill, normalized later


@dataclass
class BillData:
    # provider
    provider_name: str | None = None
    provider_address: str | None = None
    provider_phone: str | None = None

    # bill metadata
    invoice_number: str | None = None
    bill_date: str | None = None
    client_name: str | None = None
    client_id: str | None = None

    # financials
    line_items: list[LineItem] = field(default_factory=list)
    subtotal: float | None = None
    payments: float | None = None    # negative amounts / payments received
    balance_due: float | None = None
    total: float | None = None

    # provenance + trust
    source_file: str | None = None
    ocr_mean_confidence: float | None = None
    extraction_method: str | None = None   # "gemini" | "rules"
    validation_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)