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

    @classmethod
    def from_dict(cls, data: dict) -> "BillData":
        """Rebuild from to_dict() output (e.g. stored JSON). Unknown keys are ignored."""
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known and k != "line_items"}
        items = []
        for raw in data.get("line_items") or []:
            if isinstance(raw, dict):
                item_known = LineItem.__dataclass_fields__
                items.append(LineItem(**{k: v for k, v in raw.items() if k in item_known}))
        kwargs["validation_flags"] = list(kwargs.get("validation_flags") or [])
        return cls(line_items=items, **kwargs)