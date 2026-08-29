"""
Field extraction: raw bill text -> structured BillData.

Primary method: Gemini (gemini-2.5-flash) with a strict JSON schema prompt.
The LLM output is treated as untrusted: parsed defensively, validated locally.

Config: set EXTRACTION_USE_GEMINI = False to disable cloud calls entirely
(falls back to raising, until a rules-based extractor is added).
"""

from __future__ import annotations

import json
import logging
import os
import time

from dotenv import load_dotenv

from bill_ocr.errors import BillOCRError
from bill_ocr.schema import BillData, LineItem

logger = logging.getLogger(__name__)

load_dotenv()

# ── config ──────────────────────────────────────────────────────────────
EXTRACTION_USE_GEMINI = True
GEMINI_MODEL = "gemini-3.6-flash"
MAX_RETRIES = 3
# ────────────────────────────────────────────────────────────────────────


class ExtractionError(BillOCRError):
    """Raised when field extraction fails after all retries."""


PROMPT_TEMPLATE = """You are a medical bill data extraction system.
Extract the following fields from the bill text below.

Respond with ONLY a JSON object -- no markdown, no explanation, no code fences.

JSON schema (use null for anything not present in the text):
{{
  "provider_name": string or null,
  "provider_address": string or null,
  "provider_phone": string or null,
  "invoice_number": string or null,
  "bill_date": string or null,
  "client_name": string or null,
  "client_id": string or null,
  "line_items": [
    {{"description": string, "quantity": number or null, "amount": number or null, "date": string or null}}
  ],
  "subtotal": number or null,
  "payments": number or null,
  "balance_due": number or null,
  "total": number or null
}}

Rules:
- Amounts: plain numbers, no currency symbols. Negative for payments/credits.
- The text comes from OCR and may contain typos (e.g. "Involee" means "Invoice"). Interpret sensibly.
- Do NOT invent data. If a field is not in the text, use null.

BILL TEXT:
---
{bill_text}
---"""


def _get_client():
    """Lazy import + construct so the module imports fine without the SDK/key."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ExtractionError(
            "GOOGLE_API_KEY not found. Add it to the .env file in the project root."
        )
    try:
        from google import genai
    except ImportError as e:
        raise ExtractionError(
            "google-genai SDK not installed. Run: pip install google-genai"
        ) from e
    return genai.Client(api_key=api_key)


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` fences if the model added them despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def extract_fields(bill_text: str, source_file: str | None = None,
                   ocr_confidence: float | None = None) -> BillData:
    """
    Extract structured fields from raw bill text via Gemini.

    Raises:
        ExtractionError: empty input, missing key/SDK, or failure after retries.
    """
    if not bill_text or not bill_text.strip():
        raise ExtractionError("Cannot extract from empty bill text.")

    if not EXTRACTION_USE_GEMINI:
        raise ExtractionError(
            "Gemini extraction disabled (EXTRACTION_USE_GEMINI=False) "
            "and no offline extractor is configured."
        )

    from google.genai import types  # safe: _get_client verified the SDK
    client = _get_client()
    prompt = PROMPT_TEMPLATE.format(bill_text=bill_text[:15000])

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=4096,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
            raw = (response.text or "").strip()
            data = json.loads(_strip_fences(raw))
            logger.info("Extraction OK on attempt %d (%d line items)",
                        attempt, len(data.get("line_items") or []))
            return _to_bill_data(data, source_file, ocr_confidence)

        except json.JSONDecodeError as e:
            last_error = e
            logger.warning("Attempt %d: model returned invalid JSON (%s)", attempt, e)
        except Exception as e:  # network / quota / SDK errors
            last_error = e
            logger.warning("Attempt %d: API error (%s)", attempt, e)

        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)  # 2s, 4s backoff

    raise ExtractionError(
        f"Extraction failed after {MAX_RETRIES} attempts: {last_error}"
    )


def _to_bill_data(data: dict, source_file: str | None,
                  ocr_confidence: float | None) -> BillData:
    """Map the parsed JSON dict into our schema, defensively."""

    def num(value) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    items = []
    for raw_item in (data.get("line_items") or []):
        if not isinstance(raw_item, dict):
            continue
        desc = str(raw_item.get("description") or "").strip()
        if not desc:
            continue
        items.append(LineItem(
            description=desc,
            quantity=num(raw_item.get("quantity")),
            amount=num(raw_item.get("amount")),
            date=raw_item.get("date"),
        ))

    return BillData(
        provider_name=data.get("provider_name"),
        provider_address=data.get("provider_address"),
        provider_phone=data.get("provider_phone"),
        invoice_number=str(data["invoice_number"]) if data.get("invoice_number") is not None else None,
        bill_date=data.get("bill_date"),
        client_name=data.get("client_name"),
        client_id=str(data["client_id"]) if data.get("client_id") is not None else None,
        line_items=items,
        subtotal=num(data.get("subtotal")),
        payments=num(data.get("payments")),
        balance_due=num(data.get("balance_due")),
        total=num(data.get("total")),
        source_file=source_file,
        ocr_mean_confidence=ocr_confidence,
        extraction_method="gemini",
    )