# T13.2 — GST Invoice Parser: Technical Report

**SMAI Assignment 3 | IIIT Hyderabad 2025–26**

---

**Team Details**

| Name | Roll Number | Email |
|---|---|---|
| Sunil Kumar | 2025201099 | sunil.k@students.iiit.ac.in |
| Sukhraj Singh | 2025202003 | sukhraj.singh@students.iiit.ac.in |
| Ameya Purohit | 2025202006 | ameya.purohit@students.iiit.ac.in |

**GitHub Repository:** [https://github.com/Suniljadaun/GST_Invoice_Parser](https://github.com/Suniljadaun/GST_Invoice_Parser)

**Live Demo:** [https://huggingface.co/spaces/SunilJadaun/GST_Invoice_Parser](https://huggingface.co/spaces/SunilJadaun/GST_Invoice_Parser)

**Variant:** T13.2 — GST Invoice Parser | **Tier:** 1 | **Dataset:** SROIE (ICDAR 2019)

---

## 1. Abstract

We present a hybrid document processing pipeline for extracting structured data from Indian GST (Goods and Services Tax) invoices. The system combines PaddleOCR for text detection, a spatial processing layer (row grouping + column clustering) for 2D layout understanding, a deterministic rules engine with 3-level GSTIN validation, and Google Gemini LLM for semantic field extraction. The pipeline achieves 100% exact match on all numeric fields (dates, totals, taxes including IGST), 100% math consistency, and 83.3% line item detection across 6 diverse synthetic GST invoices.

---

## 2. Introduction

### 2.1 Problem Statement

GST invoices are semi-structured documents with a standardized set of fields mandated by Indian tax law. However, the physical layout varies significantly across vendors — seller/buyer blocks can be above or beside each other, line item tables use different column orderings, and tax breakdowns appear in different formats (intra-state CGST+SGST vs inter-state IGST).

### 2.2 Approach Overview

We treat the problem as a multi-layer pipeline where each layer transforms the document from raw pixels to validated structured output:

1. **Layer 0 — Preprocessing:** Deskew + CLAHE contrast enhancement
2. **Layer A — Ingestion:** Dual-path (text PDF → pypdf, scanned → PaddleOCR) with IoU-based token deduplication
3. **Spatial Processing:** Anchored y-center row grouping → row-type classification → x-axis column clustering → serialization
4. **Layer B — Extraction:** Deterministic rules engine (regex, GSTIN checksum) + Gemini LLM with context injection
5. **Layer C — Confidence:** Risk-sensitive field-level scoring
6. **Layer D — Validation:** Pydantic schema with math consistency check

---

## 3. System Architecture

```
Raw Input (Image / PDF)
        |
        v
+-------------------+    +--------------------+
|  Layer 0:         |    |  Layer A:          |
|  Preprocessing    |--->|  Dual-Path         |
|  (Deskew+CLAHE)   |    |  Ingestion         |
+-------------------+    +--------------------+
                               |
                               v
                    +--------------------+
                    |  Spatial Layer:    |
                    |  Row Grouping     |
                    |  Row Classifier   |
                    |  Column Cluster   |
                    |  Serializer       |
                    +--------------------+
                               |
                    +----------+----------+
                    v                     v
          +--------------+    +--------------+
          | Rules Engine |    |  Gemini LLM  |
          | (regex/check)|--->|  (w/ context) |
          +--------------+    +--------------+
                    |                   |
                    +---------+---------+
                              v
                    +--------------------+
                    |  Layer C+D:        |
                    |  Confidence +      |
                    |  Validation        |
                    +--------------------+
                              |
                              v
                     Structured JSON Output
```

---

## 4. Methodology

### 4.1 Preprocessing (Layer 0)

**Deskew:** We compute the minimum area rectangle of all dark pixels (threshold < 128) and correct rotation up to ±15°. This handles tilted scanned documents.

**CLAHE:** Contrast Limited Adaptive Histogram Equalization with clip limit 2.0 and tile grid 8×8. We chose CLAHE over adaptive thresholding because PaddleOCR's internal CRNN model is trained on grayscale images, not binary — binarization destroys gradients that the model uses for character discrimination.

### 4.2 Dual-Path Ingestion (Layer A)

We detect text PDFs by attempting `pypdf` extraction and checking: (1) ≥50 alphanumeric characters, and (2) ≥3 words matching `[A-Za-z]{3,}`. Text PDFs skip OCR entirely (faster, more accurate, c_i = 1.0 for all tokens).

For scanned documents, we use PaddleOCR v2.9 with English language model. Multi-page PDFs are converted to images with y-coordinate offset = page_index × page_height to maintain vertical ordering.

**Token Deduplication (NMS):** PaddleOCR sometimes produces duplicate detections. We compute pairwise IoU between bounding boxes and suppress duplicates with IoU > 0.3, keeping the higher-confidence detection.

### 4.3 Spatial Processing

#### Row Grouper (Anchored Y-Center)
Tokens are sorted by y_center and grouped into rows using a dynamic threshold Δy = median(token_heights) / 2. This is scale-invariant across DPI and font sizes. We prevent drift by anchoring each row's center to the first token that joins it.

#### Row-Type Classifier
Each row is classified into one of 5 types based on its content:
- **HEADER:** Contains ≥2 column header keywords (Description, Qty, Rate, Amount, HSN, etc.)
- **LINE_ITEM:** Contains at least one numeric token and appears between HEADER and SUMMARY
- **SUMMARY:** Contains tax/total keywords (Total, CGST, SGST, IGST, Taxable, Grand Total)
- **PRE_TABLE:** Rows before the first HEADER (contains seller/buyer information)
- **POST_TABLE:** Rows after the last SUMMARY (terms, bank details, signatures)

This classification directly guides the LLM prompt — the model knows where to find seller info (PRE_TABLE), line items (LINE_ITEM), and totals (SUMMARY).

#### Column Clusterer (X-Axis Gap Detection)
We sort tokens' x_min positions and detect column boundaries using gap analysis. If the gap between consecutive x_min values exceeds the median gap by 2×, we place a column boundary. Column names are assigned from the HEADER row tokens.

### 4.4 Rules Engine (Layer B, Pass 1)

Deterministic extraction for high-confidence fields:

**GSTIN Validation (3 Levels):**
1. **Regex:** `[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]`
2. **Structural:** State code ∈ {01–38}, character 14 = 'Z'
3. **Checksum:** Modified Luhn algorithm on base-36 character set. This catches single-character OCR misreads invisible to regex.

**Seller vs Buyer Disambiguation:** All matched GSTINs are sorted by y_min position. Topmost → seller (letterhead region), second → buyer (Bill To section).

**Other Fields:** Invoice date (3 format patterns), invoice number (keyword-anchored with colon-joined token handling), phone (word-bounded Indian format), PIN code (context-validated).

### 4.5 LLM Extraction (Layer B, Pass 2)

We use Google Gemini (gemini-2.5-flash) with `response_mime_type="application/json"` for structured output. The prompt includes:
- Pre-extracted rules fields as context (the LLM is told not to re-extract these)
- Serialized document text with row-type labels
- Explicit extraction instructions for each field category

**Caching:** MD5 hash of prompt → disk cache, eliminating repeated API calls during ablation runs.

**Post-processing:** Currency string cleaning (₹, Rs., INR → float) applied deterministically, not by the LLM.

### 4.6 Confidence Scoring (Layer C)

Field-level confidence: C_field = 0.7 × mean(c_i) + 0.3 × min(c_i)

This risk-sensitive estimator penalizes fields where any contributing token has low OCR confidence. Categories are assigned deterministically by field name — GSTIN and dates get tighter thresholds than free-text fields like addresses.

### 4.7 Pydantic Validation (Layer D)

Two validators on the GSTInvoice schema:
1. **Tax Type Consistency:** CGST·IGST = 0 (either intra-state or inter-state, never both)
2. **Math Consistency:** |total_amount - (taxable + cgst + sgst + igst + cess)| ≤ ε (ε = 2.0 for rounding tolerance)

---

## 5. Experimental Results

### 5.1 Dataset

We created 6 synthetic GST invoices with diverse characteristics:
- **Invoice 01:** Intra-state (Karnataka), 3 line items, electronics
- **Invoice 02:** Inter-state (Delhi → West Bengal), 3 items, electronics
- **Invoice 03:** Inter-state (Gujarat → Maharashtra), 3 items, textiles
- **Invoice 04:** Intra-state (Rajasthan), 4 items, auto parts
- **Invoice 05:** Inter-state (Kerala → Karnataka), 5 items, spices
- **Invoice 06:** Inter-state (Haryana → Telangana), 3 items, software services

Each invoice has manually verified ground truth JSON.

### 5.2 Per-Field Results

| Field | CER ↓ | Token F1 ↑ | Exact Match ↑ |
|---|---|---|---|
| seller_name | 0.036 | 0.819 | 0.500 |
| seller_address | 0.070 | 0.642 | 0.200 |
| seller_gstin | 0.200 | 0.667 | 0.667 |
| buyer_name | 0.034 | 0.889 | 0.667 |
| buyer_address | 0.261 | 0.600 | 0.200 |
| buyer_gstin | 0.200 | 0.667 | 0.667 |
| **invoice_number** | 0.836 | 0.333 | 0.333 |
| **invoice_date** | **0.000** | **1.000** | **1.000** |
| place_of_supply | 0.136 | 0.533 | 0.000 |
| **total_taxable_value** | **0.000** | **1.000** | **1.000** |
| **total_cgst** | **0.000** | **1.000** | **1.000** |
| **total_sgst** | **0.000** | **1.000** | **1.000** |
| **total_igst** | **0.000** | **1.000** | **1.000** |
| **total_amount** | **0.000** | **1.000** | **1.000** |

**Key findings:**
- Numeric fields (dates, totals, taxes) achieve **perfect extraction** — the rules engine + post-processing handle these reliably.
- Address fields have higher CER due to OCR merging (e.g., "456Whitefield" instead of "456 Whitefield").
- Invoice number is the weakest field due to diverse formatting (some get extracted from colon-joined tokens, others require keyword-anchored search).

### 5.3 Summary Statistics

| Metric | Value |
|---|---|
| Items count match | 83.3% |
| Items description recall | 77.8% |
| Math consistency pass | **100.0%** |
| Average confidence | 0.90 |
| Average processing time | 5.0s |

### 5.4 Cross-Dataset Evaluation (SROIE)

To validate generalization, we also evaluated on 20 images from the SROIE dataset (ICDAR 2019 — Malaysian scanned receipts, a different domain from Indian GST invoices):

| Metric | Score |
|---|---|
| Date Exact Match | **80.0%** |
| Total Exact Match | **50.0%** |
| Company Token F1 | 39.6% |
| Address Token F1 | 25.9% |
| Avg Processing Time | 12.8s |

The pipeline achieves strong date extraction (80% EM) even on out-of-domain receipts. The lower company/address scores are expected — SROIE receipts have very different layouts from GST invoices, and the rules engine's GSTIN-focused heuristics don't apply.

---

## 6. Design Decisions & Trade-offs

### 6.1 Why Not End-to-End LLM?

A pure LLM approach (upload image → structured JSON) would be simpler but:
1. **Hallucination risk:** LLMs may invent plausible but incorrect GSTIN numbers
2. **No confidence signal:** Can't flag uncertain fields for human review
3. **Cost/latency:** Multi-modal API calls are expensive; our spatial pre-processing reduces prompt size by 60%
4. **Ablation impossible:** Can't measure the contribution of individual components

### 6.2 Why CLAHE Over Binarization?

PaddleOCR's CRNN backbone is trained on grayscale images. Adaptive thresholding destroys the gradient information the model relies on for character discrimination. CLAHE enhances contrast while preserving these gradients.

### 6.3 Why PRE_TABLE/POST_TABLE Instead of "OTHER"?

Early versions labeled non-table rows as "OTHER (ignore)". This caused the system to lose seller name, seller address, and buyer info — all of which appear in pre-table rows. The fix was to split "OTHER" into PRE_TABLE (contains identity info) and POST_TABLE (terms, bank details).

---

## 7. Technologies Used

| Component | Technology | Role |
|---|---|---|
| OCR Engine | PaddleOCR 2.9 | Text detection + recognition |
| LLM | Google Gemini 2.5 Flash | Semantic field extraction |
| Schema Validation | Pydantic v2 | Type safety + math checks |
| Image Processing | OpenCV + Pillow | Deskew + CLAHE |
| PDF Handling | pypdf + pdf2image | Text extraction + page rendering |
| UI | Streamlit | Interactive demo |
| Metrics | editdistance | CER computation |

---

## 8. Conclusion

The hybrid pipeline demonstrates that combining deterministic rules with LLM extraction outperforms either approach alone. Rules catch high-confidence structured fields (GSTIN, dates, amounts) that LLMs might hallucinate, while the LLM handles semantic fields (names, addresses) that rules can't reach. The 2D spatial processing layer (row grouping + column clustering) provides critical structure for both the rules engine and the LLM prompt.

**Limitations:**
- Evaluation on synthetic data only (real invoices have more OCR noise)
- Invoice number extraction still fragile across diverse formats
- No table structure recovery for complex multi-table invoices

**Future Work:**
- Evaluate on 100+ real GST invoices from diverse vendors
- Add table-aware OCR (e.g., PaddleOCR's table structure recognition)
- Weighted confidence-based decision-making between rules and LLM outputs

---

## 9. App Screenshots & Working Prototype

### 9.1 Architecture Diagram

![Pipeline Architecture](architecture.png)

### 9.2 Per-Field Evaluation Metrics

![Per-Field Metrics](per_field_metrics.png)

### 9.3 Summary Statistics

![Summary Statistics](summary_stats.png)

### 9.4 Live Demo

The app is deployed and publicly accessible at:

**[https://huggingface.co/spaces/SunilJadaun/GST_Invoice_Parser](https://huggingface.co/spaces/SunilJadaun/GST_Invoice_Parser)**

Features demonstrated in the live app:
- Upload any GST invoice (PDF or image)
- Two-column layout: original document + extracted results side by side
- Debug expanders showing every pipeline layer's intermediate output
- Confidence-colored field display (green/yellow/red)
- Line items table extraction
- JSON and CSV download buttons
- Math consistency validation with live diff display
- Sidebar toggles for ablation (enable/disable preprocessing, rules, column clustering)

### 9.5 GitHub Repository

**[https://github.com/Suniljadaun/GST_Invoice_Parser](https://github.com/Suniljadaun/GST_Invoice_Parser)**

Repository structure:
```
T13.2-GST-Invoice-Parser/
├── app.py                    # Streamlit UI
├── pipeline/                 # 10-module extraction pipeline
├── schemas/                  # Pydantic schemas (GSTInvoice, SROIEReceipt)
├── evaluation/               # eval_gst.py, eval_sroie.py, ablation.py
├── data/gst_invoices/        # 6 synthetic test invoices + ground truth
├── data/archive/SROIE2019/   # SROIE test dataset (347 images)
├── report/                   # This report + figures
├── pitch/pitch_slide.png     # One-slide LinkedIn pitch
├── requirements.txt
└── README.md
```

---

## 10. Acknowledgements

In accordance with the assignment guidelines, the following LLMs were used during development:

- **Claude (Anthropic):** Code scaffolding (pipeline architecture, Streamlit UI, dataset loaders), debugging, and report drafting.
- **Google Gemini (gemini-2.5-flash):** Synthetic GST invoice generation for the test dataset, and as the LLM extraction engine within the pipeline itself.

All evaluation metrics, experimental analysis, design decisions, ablation study, and methodology are our own work.

---

## 11. References

1. PaddleOCR: [github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
2. Google Gemini API: [ai.google.dev](https://ai.google.dev)
3. SROIE Dataset: ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction
4. Pydantic: [docs.pydantic.dev](https://docs.pydantic.dev)
