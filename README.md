# Medical Bill OCR Pipeline

Production-grade pipeline that turns medical bills — scanned photos, scanned
PDFs, and digital PDFs — into validated, review-ready Excel/CSV data.

**Design principle:** the system flags what it cannot verify instead of
guessing. Every extracted bill passes local arithmetic validation; uncertain
rows are marked `NEEDS_REVIEW` with specific reasons.

## Architecture
input file
└─ ingest: classify ──→ digital PDF ──→ read text layer (exact, no OCR)
├─→ scanned PDF ──→ rasterize 300 DPI ─┐
└─→ image ─────────────────────────────┤
▼
preprocess (OpenCV): upscale → denoise
→ deskew → adaptive threshold
▼
OCR (Tesseract 5 LSTM) + per-word confidence
▼
extract (Gemini 2.5 Flash, temperature 0, strict JSON schema)
▼
validate (local, deterministic): line items must reconcile
against printed totals; dates parseable; confidence checks
▼
export: two-sheet Excel (Bills / Line Items) + CSV pair,
NEEDS_REVIEW column first


The LLM is treated as an untrusted component: its output is parsed
defensively and re-verified by local math before anything reaches Excel.

## Setup (Windows)

1. **Install Tesseract** (standalone program, not pip):
   https://github.com/UB-Mannheim/tesseract/wiki — 64-bit installer,
   default components. Verify: `tesseract --version`

2. **Project setup:**
```powershell
   git clone https://github.com/NIKHIL-956653/medical-bill-ocr.git
   cd medical-bill-ocr
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip   # required: old pip cannot editable-install
   pip install -e .
```

3. **API key** — create `.env` in the project root:

GOOGLE_API_KEY=your_key_from_aistudio.google.com/apikey


## Usage

```powershell
python -m bill_ocr process --input data/input --output data/output/results.xlsx
```

Options:
- `--debug-dir artifacts/debug` — writes every preprocessing stage image
  (grayscale, denoised, deskewed, binarized) for inspection when OCR misreads
- `-v` — debug logging

Output: `results.xlsx` (sheets: **Bills**, **Line Items**) plus
`results_bills.csv` / `results_line_items.csv`.

## Review web UI

A local web app for checking and correcting extractions before they go to Excel.

```powershell
pip install -e ".[web]"
python -m bill_ocr serve            # opens on http://127.0.0.1:8000
```

- **Upload** PDFs or images (drag and drop, several at once). Each runs through
  the same pipeline as the CLI in the background.
- **Review side by side**: the page image on the left, extracted fields on the
  right. Clicking a field highlights where that value appears on the page,
  using the OCR word boxes (or the PDF text layer for digital PDFs). No
  highlight means the value is not on the page, which is a hint it is wrong.
- **Correct** any field or line item. Fields that differ from the extraction
  turn yellow, the line-item sum is re-checked live, and saving re-runs the
  validation checks. A toggle shades low-confidence OCR words.
- **Export** reviewed bills (or all of them) to the same two-sheet workbook.
- **Accuracy**: every save keeps both the original extraction and the
  reviewer's version. The Accuracy view compares them field by field, by week,
  and by model, so the effect of prompt, model or preprocessing changes is
  measurable. Same report on the command line:

```powershell
python -m bill_ocr accuracy
```

Everything is stored under `data/review/` (SQLite database, uploads, page
images), which is git-ignored. Every save is also kept as a revision for
auditing. The server has no login and binds to localhost by default; only use
`--host 0.0.0.0` on a trusted network.

## Trust model

| Check | Action on failure |
|---|---|
| Line items sum == printed subtotal/total (±$0.05) | `totals_mismatch` flag → NEEDS_REVIEW |
| OCR mean confidence >= 65 | `low_ocr_confidence` flag |
| Provider / invoice # / date present | `missing_*` flags |
| Date parseable (8 common formats) | `unparseable_date` flag |
| Per-file failure isolation | one corrupt file never aborts the batch |

## Privacy

- Bills live only in `data/` — hard-blocked from git
- `.env` (API key) — git-ignored
- Cloud extraction (Gemini) is approved by the client; the extraction layer
  is a single module (`extract.py`) and can be swapped for a local model
  without touching the rest of the pipeline

## Tests

```powershell
pytest tests/ -v
```

45 tests — schema, validation math, defensive JSON parsing, export,
ingest routing, value location, accuracy metrics, review storage, and the
web API — all offline, zero API calls. Install test dependencies with
`pip install -e ".[dev]"`. Tests that run real OCR skip when Tesseract is
not installed.

## Stack

Python 3.10+ · pymupdf · pdfplumber · OpenCV · Tesseract 5 (pytesseract) ·
Gemini 2.5 Flash (google-genai) · pandas + openpyxl · FastAPI + SQLite (review UI) · pytest
