"""
Find where an extracted value appears on the page.

Given a value ("57.69", "Test Clinic", "11/07/2024") and the word boxes of each
page, return the boxes of the words that make it up. The review UI draws these
over the page image so a reviewer can check a value against its source.

Words are stored compactly as [text, confidence, x, y, w, h] with coordinates
as fractions of the page size.
"""

from __future__ import annotations

from difflib import SequenceMatcher

MONEY_EPS = 0.005
FUZZY_MIN_RATIO = 0.8      # tolerate OCR typos like "Clinlc" for "Clinic"
FUZZY_MIN_LEN = 4          # short tokens must match exactly
MAX_MATCHES = 20

_STRIP = " \t,.:;()[]{}$#*\"'|"


def _norm(token: str) -> str:
    return token.strip(_STRIP).casefold()


def _as_number(text) -> float | None:
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return float(text)
    cleaned = str(text).strip().replace("$", "").replace(",", "").strip("()")
    if not cleaned or not any(c.isdigit() for c in cleaned):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _token_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) < FUZZY_MIN_LEN:
        return False
    return SequenceMatcher(None, a, b).ratio() >= FUZZY_MIN_RATIO


def _box(page: int, word: list) -> dict:
    return {"page": page, "x": word[2], "y": word[3], "w": word[4], "h": word[5]}


def locate(value, pages: list[list[list]]) -> list[dict]:
    """
    Return boxes ({page, x, y, w, h}) for every place `value` appears.
    Numbers match by numeric value (ignores $, commas, sign); text matches as a
    consecutive run of words, case-insensitive, with light fuzzy matching.
    """
    if value is None or str(value).strip() == "":
        return []

    number = _as_number(value)
    if number is not None and len(str(value).split()) == 1:
        return _locate_number(number, pages)
    return _locate_text(str(value), pages)


def _locate_number(number: float, pages: list[list[list]]) -> list[dict]:
    boxes = []
    for p, words in enumerate(pages):
        for word in words:
            n = _as_number(word[0])
            if n is not None and abs(abs(n) - abs(number)) <= MONEY_EPS:
                boxes.append(_box(p, word))
                if len(boxes) >= MAX_MATCHES:
                    return boxes
    return boxes


def _locate_text(text: str, pages: list[list[list]]) -> list[dict]:
    target = [t for t in (_norm(tok) for tok in text.split()) if t]
    if not target:
        return []

    boxes = []
    matches = 0
    for p, words in enumerate(pages):
        normed = [_norm(w[0]) for w in words]
        i = 0
        while i <= len(words) - len(target):
            if all(_token_match(normed[i + k], target[k]) for k in range(len(target))):
                boxes.extend(_box(p, words[i + k]) for k in range(len(target)))
                matches += 1
                if matches >= MAX_MATCHES:
                    return boxes
                i += len(target)
            else:
                i += 1
    return boxes
