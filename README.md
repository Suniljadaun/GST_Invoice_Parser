---
title: GST Invoice Parser
emoji: 🧾
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
---

# T13.2 — GST Invoice Parser

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/SunilJadaun/GST_Invoice_Parser)  
**Live Demo:** [https://huggingface.co/spaces/SunilJadaun/GST_Invoice_Parser](https://huggingface.co/spaces/SunilJadaun/GST_Invoice_Parser)

**SMAI Assignment 3 | IIIT Hyderabad 2025–26**

A hybrid OCR + Rules + LLM pipeline for extracting structured data from Indian GST invoices.

## Architecture

```
Image/PDF → Preprocessing → Dual-Path OCR → Row Grouping → Row Classification
    → Column Clustering → Serialization → Rules Engine → Gemini LLM
    → Confidence Scoring → Pydantic Validation → Structured JSON
```

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set API key
cp .env.example .env
# Edit .env with your free Gemini API key from https://aistudio.google.com/apikey

# 3. Run the app
streamlit run app.py
```

## Project Structure

```
T13.2-GST-Invoice-Parser/
├── app.py                          # Streamlit UI
├── pipeline/
│   ├── __init__.py                 # Orchestrator
│   ├── preprocessing.py            # Deskew + CLAHE
│   ├── ingestion.py                # Dual-path (pypdf / PaddleOCR) + NMS dedup
│   ├── row_grouper.py              # Anchored y-center algorithm
│   ├── row_classifier.py           # HEADER/LINE_ITEM/SUMMARY/PRE_TABLE/POST_TABLE
│   ├── column_clusterer.py         # X-axis gap detection
│   ├── serializer.py               # Named columns or x-coordinate fallback
│   ├── rules.py                    # 3-level GSTIN validation + keyword extraction
│   ├── llm_extractor.py            # Gemini structured output + MD5 cache
│   ├── confidence.py               # Risk-sensitive field-level scoring
│   └── validator.py                # Pydantic schema validation
├── schemas/
│   ├── gst_invoice.py              # GSTInvoice + LineItem (20+ fields, 2 validators)
│   └── sroie_receipt.py            # SROIEReceipt (4 fields)
├── evaluation/
│   ├── __init__.py                 # Metrics: CER, Token F1, Exact Match
│   ├── eval_gst.py                 # GST invoice evaluation runner
│   ├── eval_sroie.py               # SROIE dataset evaluation runner
│   └── ablation.py                 # 8-variant ablation table generator
├── data/
│   ├── gst_invoices/               # 6 synthetic test invoices + ground truth
│   ├── sroie_test/                 # SROIE test images (download separately)
│   └── llm_cache/                  # MD5-cached Gemini responses
├── report/
│   ├── report.pdf                  # Technical report (PDF)
│   ├── report.md                   # Report source (Markdown)
│   ├── per_field_metrics.png       # Per-field CER/F1/EM chart
│   ├── summary_stats.png           # Pipeline performance summary
│   └── architecture.png            # Architecture diagram
├── pitch/
│   ├── pitch_slide.png             # One-slide pitch (PNG)
│   └── pitch.md                    # Viva presentation points
├── requirements.txt
├── packages.txt                    # System packages for deployment
├── .env.example
└── test_pipeline.py                # Quick end-to-end test
```

## Results (6 GST Invoices)

| Metric | Value |
|---|---|
| Date Exact Match | **100%** |
| Total Amount EM | **100%** |
| All Tax Fields EM | **100%** |
| Math Consistency | **100%** |
| Line Items Recall | **83.3%** |
| Avg Processing Time | **5.0s** |
| Avg Confidence | **0.90** |

## Key Features

- **3-Level GSTIN Validation:** Regex → Structural → Checksum (modified Luhn base-36)
- **Seller/Buyer Disambiguation:** Y-position based spatial sorting
- **Row-Type Classification:** PRE_TABLE / HEADER / LINE_ITEM / SUMMARY / POST_TABLE
- **Dual-Path Ingestion:** Text PDF (pypdf) vs Scanned (PaddleOCR)
- **Token NMS:** IoU-based deduplication for duplicate OCR detections
- **CLAHE Preprocessing:** Preserves gradients for PaddleOCR (vs binarization)
- **LLM Context Injection:** Rules fields fed as verified context to Gemini
- **MD5 Disk Cache:** Avoids repeated API calls during ablation
- **Math Validation:** |total - components| ≤ ε with rounding tolerance

## SROIE Dataset Setup

The SROIE evaluation requires the Kaggle dataset (1,000 receipt images):

```bash
# Option 1: Automated download (requires Kaggle API key)
pip install kaggle
./download_sroie.sh

# Option 2: Manual download
# 1. Download from https://www.kaggle.com/datasets/urbikn/sroie-datasetv2
# 2. Extract test images to data/sroie_test/img/
# 3. Extract test keys to data/sroie_test/key/
```

## Running Evaluation

```bash
# Run on GST invoices (included, no download needed)
python3 evaluation/eval_gst.py

# Run ablation study (requires SROIE data — see above)
python3 evaluation/ablation.py --data-dir data/sroie_test --max-samples 20
```

