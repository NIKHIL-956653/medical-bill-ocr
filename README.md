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

23 tests — schema, validation math, defensive JSON parsing, export,
ingest routing — all offline, zero API calls.

## Stack

Python 3.10+ · pymupdf · pdfplumber · OpenCV · Tesseract 5 (pytesseract) ·
Gemini 2.5 Flash (google-genai) · pandas + openpyxl · pytest
