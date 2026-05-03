"""
🧾 GST Invoice Parser — T13.2 SMAI Assignment
IIIT Hyderabad · Hybrid OCR + Rules + LLM + Validation

Streamlit app with:
  - Upload PDF/image
  - Two-column layout: original doc + pipeline results
  - Debug expanders for every pipeline layer
  - Confidence-colored data editor
  - JSON + CSV download
"""

import json
import os
import sys
import tempfile
import logging

import streamlit as st
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import run_pipeline, PipelineResult
from pipeline.confidence import get_confidence_color

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="GST Invoice Parser — T13.2",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title("🧾 GST Invoice Parser")
st.caption(
    "IIIT Hyderabad · SMAI Assignment 3 · T13.2 | "
    "Hybrid OCR + Rules + LLM + Validation"
)

# ──────────────────────────────────────────────
# Sidebar settings
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Settings")
    mode = st.selectbox("Mode", ["GST Invoice", "SROIE Receipt"],
                        help="GST: full 20+ field extraction. SROIE: 4 fields only.")
    use_preprocessing = st.checkbox("Image Preprocessing (deskew + CLAHE)", value=True)
    use_columns = st.checkbox("Column Clustering", value=True)
    use_rules = st.checkbox("Rules Pass (regex)", value=True)
    inject_context = st.checkbox("Context Injection to LLM", value=True)
    use_cache = st.checkbox("Use LLM Cache", value=True)

    st.divider()
    st.caption("Set GOOGLE_API_KEY in .env or environment")
    api_key_input = st.text_input("API Key (optional override)", type="password")
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input

# ──────────────────────────────────────────────
# File upload
# ──────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload PDF or Image",
    type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "webp"],
    help="Supports scanned PDFs, native text PDFs, and images",
)

if not uploaded:
    st.info("👆 Upload an invoice to get started")
    st.stop()

# Save uploaded file to temp location
suffix = os.path.splitext(uploaded.name)[1]
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

# ──────────────────────────────────────────────
# Run pipeline
# ──────────────────────────────────────────────
pipeline_mode = "sroie" if mode == "SROIE Receipt" else "gst"

with st.spinner("🔄 Running extraction pipeline..."):
    result: PipelineResult = run_pipeline(
        tmp_path,
        mode=pipeline_mode,
        use_columns=use_columns,
        use_rules=use_rules,
        use_preprocessing=use_preprocessing,
        inject_context=inject_context,
        use_cache=use_cache,
    )

# Clean up temp file
try:
    os.unlink(tmp_path)
except OSError:
    pass

# ──────────────────────────────────────────────
# Two-column layout
# ──────────────────────────────────────────────
col_left, col_right = st.columns([2, 3])

# ── LEFT: Original document + debug expanders ──
with col_left:
    st.subheader("📄 Original Document")
    if suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        uploaded.seek(0)
        st.image(uploaded, width="stretch")
    else:
        st.info(f"📎 Uploaded: {uploaded.name} ({suffix.upper()} file)")

    st.divider()

    # Debug: Spatial OCR Rows
    with st.expander("🔍 Spatial OCR Rows", expanded=False):
        if result.classified_rows:
            for i, (rt, row_tokens) in enumerate(result.classified_rows):
                tokens_str = " | ".join(
                    f'"{t.text}"' for t in row_tokens
                )
                st.text(f"Row {i} [{rt}]: {tokens_str}")
        else:
            st.write("No rows detected")

    # Debug: Column Detection
    with st.expander("📊 Column Detection", expanded=False):
        if result.column_info:
            boundaries, col_names = result.column_info
            st.write("**Boundaries:**", [f"x={b:.0f}" for b in boundaries])
            st.write("**Column Names:**", col_names)
        else:
            st.write("Column detection not available (using x-coordinate fallback)")

    # Debug: Raw LLM Response
    with st.expander("🤖 Raw LLM Response", expanded=False):
        st.json(result.llm_raw)

    # Debug: Confidence Scores
    with st.expander("📈 Confidence Scores", expanded=False):
        if result.confidences:
            conf_data = []
            for field, score in sorted(result.confidences.items()):
                color = get_confidence_color(score)
                conf_data.append({
                    "Field": field,
                    "Confidence": f"{score:.2f}",
                    "Status": color,
                })
            st.table(pd.DataFrame(conf_data))

            st.metric("Invoice Confidence (C_invoice)",
                      f"{result.invoice_confidence:.2f}")
        else:
            st.write("No confidence scores available")

# ── RIGHT: Pipeline status + extracted data ──
with col_right:
    st.subheader("📋 Pipeline Status")

    # Pipeline status trail
    for step in result.steps:
        st.write(step)

    st.caption(f"⏱️ Processing time: {result.processing_time:.2f}s")

    st.divider()

    # Extracted data
    st.subheader("📝 Extracted Invoice Data")

    if result.extracted:
        # Build display data with confidence colors
        display_data = {}
        for key, value in result.extracted.items():
            if key == "items":
                continue  # show separately
            conf = result.confidences.get(key, 0.0)
            color = get_confidence_color(conf)
            display_data[key] = {
                "Value": str(value) if value is not None else "—",
                "Confidence": f"{color} {conf:.2f}",
            }

        st.dataframe(
            pd.DataFrame(display_data).T,
            width="stretch",
        )

        # Line items table
        items = result.extracted.get("items", [])
        if items:
            st.subheader("📦 Line Items")
            items_df = pd.DataFrame(items)
            st.dataframe(items_df, width="stretch")

    st.divider()

    # Download buttons
    dcol1, dcol2 = st.columns(2)

    with dcol1:
        json_str = json.dumps(result.extracted, indent=2, default=str)
        st.download_button(
            "📥 Download JSON",
            data=json_str,
            file_name="invoice_extracted.json",
            mime="application/json",
        )

    with dcol2:
        # CSV: flatten the main fields
        flat = {k: v for k, v in result.extracted.items() if k != "items"}
        csv_str = pd.DataFrame([flat]).to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            data=csv_str,
            file_name="invoice_extracted.csv",
            mime="text/csv",
        )

    # Validation details
    with st.expander("📐 Validation Details", expanded=False):
        if result.validation_errors:
            st.warning("Validation issues found:")
            for err in result.validation_errors:
                st.write(f"- {err}")
        else:
            st.success("All validations passed ✅")

        if pipeline_mode == "gst" and result.extracted:
            st.write("**Math check:**")
            t = result.extracted
            taxable = t.get("total_taxable_value") or 0
            cgst = t.get("total_cgst") or 0
            sgst = t.get("total_sgst") or 0
            igst = t.get("total_igst") or 0
            cess = t.get("total_cess") or 0
            total = t.get("total_amount") or 0
            expected = taxable + cgst + sgst + igst + cess
            diff = abs(expected - total) if total else 0
            st.write(
                f"  Components: {taxable} + {cgst} + {sgst} + {igst} + {cess} "
                f"= **{expected:.2f}**"
            )
            st.write(f"  Parsed total: **{total}**")
            st.write(f"  Difference: **{diff:.2f}** (ε = 2.0)")
            if diff <= 2.0:
                st.success(f"|{expected:.2f} - {total}| = {diff:.2f} ≤ 2.0 ✅")
            else:
                st.error(f"|{expected:.2f} - {total}| = {diff:.2f} > 2.0 ❌")
