"""
Extraction accuracy, measured against human corrections.

Every reviewed bill has two snapshots: what the pipeline extracted, and what a
reviewer saved as correct. Comparing them field by field gives a running
accuracy score per field, per model, and per week -- so changes to prompts,
models or preprocessing can be judged by numbers instead of impressions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

TEXT_FIELDS = ["provider_name", "provider_address", "provider_phone",
               "invoice_number", "bill_date", "client_name", "client_id"]
MONEY_FIELDS = ["subtotal", "payments", "balance_due", "total"]
HEADER_FIELDS = TEXT_FIELDS + MONEY_FIELDS
ITEM_TEXT_FIELDS = ["description", "date"]
ITEM_NUMBER_FIELDS = ["quantity", "amount"]
ITEM_FIELDS = ITEM_TEXT_FIELDS + ITEM_NUMBER_FIELDS

NUMBER_EPS = 0.005


@dataclass
class FieldResult:
    field: str            # e.g. "total", "line_items.amount", "line_item_count"
    correct: bool
    extracted: object
    final: object


@dataclass
class ReviewRecord:
    extracted: dict
    final: dict
    reviewed_at: str      # ISO timestamp
    model: str | None = None


def _blank(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _text_equal(a, b) -> bool:
    if _blank(a) and _blank(b):
        return True
    if _blank(a) or _blank(b):
        return False
    return " ".join(str(a).split()).casefold() == " ".join(str(b).split()).casefold()


def _number_equal(a, b) -> bool:
    if _blank(a) and _blank(b):
        return True
    try:
        return abs(float(a) - float(b)) <= NUMBER_EPS
    except (TypeError, ValueError):
        return False


def compare_bills(extracted: dict, final: dict) -> list[FieldResult]:
    """Field-by-field comparison of an extraction against its reviewed version."""
    results = []
    for f in TEXT_FIELDS:
        results.append(FieldResult(f, _text_equal(extracted.get(f), final.get(f)),
                                   extracted.get(f), final.get(f)))
    for f in MONEY_FIELDS:
        results.append(FieldResult(f, _number_equal(extracted.get(f), final.get(f)),
                                   extracted.get(f), final.get(f)))

    ext_items = extracted.get("line_items") or []
    fin_items = final.get("line_items") or []
    results.append(FieldResult("line_item_count", len(ext_items) == len(fin_items),
                               len(ext_items), len(fin_items)))

    # items are aligned by position; an added or deleted row counts as wrong
    for i in range(max(len(ext_items), len(fin_items))):
        e = ext_items[i] if i < len(ext_items) else {}
        g = fin_items[i] if i < len(fin_items) else {}
        missing = i >= len(ext_items) or i >= len(fin_items)
        for f in ITEM_FIELDS:
            equal = _text_equal if f in ITEM_TEXT_FIELDS else _number_equal
            ok = (not missing) and equal(e.get(f), g.get(f))
            results.append(FieldResult(f"line_items.{f}", ok, e.get(f), g.get(f)))
    return results


def _rate(correct: int, total: int) -> float | None:
    return round(correct / total, 4) if total else None


def _week(ts: str) -> str:
    try:
        year, week, _ = datetime.fromisoformat(ts).isocalendar()
        return f"{year}-W{week:02d}"
    except (TypeError, ValueError):
        return "unknown"


def accuracy_report(records: list[ReviewRecord]) -> dict:
    """
    Aggregate accuracy over reviewed bills.

    Returns a JSON-ready dict with overall numbers, a per-field table, and
    breakdowns by ISO week and by extraction model.
    """
    per_field: dict[str, list[int]] = defaultdict(lambda: [0, 0])     # correct, total
    by_week: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # fc, ft, perfect, docs
    by_model: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    total_correct = total_fields = perfect_docs = 0

    for rec in records:
        results = compare_bills(rec.extracted, rec.final)
        correct = sum(r.correct for r in results)
        perfect = int(correct == len(results))
        total_correct += correct
        total_fields += len(results)
        perfect_docs += perfect
        for r in results:
            per_field[r.field][0] += int(r.correct)
            per_field[r.field][1] += 1
        for bucket in (by_week[_week(rec.reviewed_at)], by_model[rec.model or "unknown"]):
            bucket[0] += correct
            bucket[1] += len(results)
            bucket[2] += perfect
            bucket[3] += 1

    order = HEADER_FIELDS + ["line_item_count"] + [f"line_items.{f}" for f in ITEM_FIELDS]
    fields = [
        {"field": f, "correct": per_field[f][0], "total": per_field[f][1],
         "accuracy": _rate(*per_field[f])}
        for f in order if per_field[f][1]
    ]

    def breakdown(groups: dict[str, list[int]], key: str) -> list[dict]:
        return [
            {key: k, "documents": v[3], "field_accuracy": _rate(v[0], v[1]),
             "perfect_rate": _rate(v[2], v[3])}
            for k, v in sorted(groups.items())
        ]

    return {
        "documents": len(records),
        "field_accuracy": _rate(total_correct, total_fields),
        "perfect_documents": perfect_docs,
        "perfect_rate": _rate(perfect_docs, len(records)),
        "fields": fields,
        "by_week": breakdown(by_week, "week"),
        "by_model": breakdown(by_model, "model"),
    }
