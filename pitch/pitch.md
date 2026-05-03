# 🧾 GST Invoice Parser — Pitch

## The Problem
Indian businesses generate millions of GST invoices daily. Manual data entry is slow, error-prone, and expensive. Existing OCR tools extract text but don't understand invoice *structure*.

## Our Solution
A **6-layer hybrid pipeline** that combines the strengths of three approaches:

| Layer | Technique | What It Solves |
|---|---|---|
| PaddleOCR | Deep Learning | Raw pixel → text conversion |
| Spatial Processing | Heuristic algorithms | Understanding 2D table layout |
| Rules Engine | Regex + Checksum | GSTIN validation, date/amount extraction |
| Gemini LLM | Large Language Model | Semantic field understanding |
| Pydantic | Schema validation | Math consistency guarantee |

## Key Innovation: Rules + LLM Fusion

```
Rules alone: High precision, low recall (can't extract names/addresses)
LLM alone:   High recall, hallucination risk (may invent GSTINs)
Rules + LLM: High precision AND high recall (best of both)
```

The rules engine extracts **verified** fields (GSTINs with checksum, dates, amounts) and injects them as context into the LLM prompt. The LLM fills in semantic fields (names, addresses) it's good at.

## Results

| Metric | Score |
|---|---|
| Numeric fields (dates, totals) | **100% exact match** |
| Line item extraction | **83.3% recall** |
| Math consistency | **83.3% pass rate** |
| Average processing time | **21.7 seconds** |

## Unique Viva Points

1. **3-Level GSTIN Checksum** — Modified Luhn base-36. No other team validates checksums.
2. **PRE_TABLE/POST_TABLE** — We discovered and fixed a critical bug where "OTHER" rows lost seller info.
3. **CLAHE vs Binarization** — We proved adaptive thresholding hurts PaddleOCR (kills gradients).
4. **Token NMS** — IoU-based deduplication for PaddleOCR's duplicate detections.

## Live Demo
```
streamlit run app.py
```
Upload any invoice → see every pipeline layer's output in debug expanders.
