"""
Pipeline Orchestrator

Runs the full extraction pipeline:
  Layer 0: Preprocessing (deskew + CLAHE)
  Layer A: Dual-path ingestion (pypdf vs PaddleOCR)
  Pre-Processor: Row grouping → Row classification → Column clustering → Serialization
  Layer B: Rules pass + LLM extraction
  Layer C: Confidence scoring
  Layer D: Pydantic validation
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .ingestion import Token, ingest
from .row_grouper import group_into_rows
from .row_classifier import classify_rows
from .column_clusterer import cluster_columns
from .serializer import serialize
from .rules import run_rules_pass
from .llm_extractor import extract_with_llm
from .confidence import compute_all_confidences, compute_invoice_confidence
from .validator import validate_gst_invoice, validate_sroie_receipt

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result from the extraction pipeline, used by the Streamlit UI."""

    # Pipeline metadata
    input_method: str = ""            # "text_pdf" or "paddleocr"
    processing_time: float = 0.0      # seconds

    # Layer outputs (for debug expanders)
    tokens: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    classified_rows: list = field(default_factory=list)
    column_info: Optional[tuple] = None
    serialized_text: str = ""
    rules_fields: dict = field(default_factory=dict)
    llm_raw: dict = field(default_factory=dict)

    # Final output
    extracted: dict = field(default_factory=dict)
    confidences: dict = field(default_factory=dict)
    invoice_confidence: float = 0.0
    validation_errors: list = field(default_factory=list)
    validated_invoice: object = None

    # Status trail
    steps: list = field(default_factory=list)


def run_pipeline(
    file_path: str,
    mode: str = "gst",
    use_columns: bool = True,
    use_rules: bool = True,
    use_preprocessing: bool = True,
    inject_context: bool = True,
    use_cache: bool = True,
) -> PipelineResult:
    """
    Run the full extraction pipeline.

    Args:
        file_path: path to PDF or image file
        mode: "gst" or "sroie"
        use_columns: whether to run column clustering
        use_rules: whether to run regex rules pass
        use_preprocessing: whether to apply deskew + CLAHE
        inject_context: whether to inject rules fields into LLM prompt
        use_cache: whether to use LLM response cache

    Returns:
        PipelineResult with all intermediate outputs for UI display
    """
    result = PipelineResult()
    start = time.time()

    try:
        # ── Layer A: Ingestion ──
        tokens, input_method = ingest(file_path, apply_preprocessing=use_preprocessing)
        result.tokens = tokens
        result.input_method = input_method
        result.steps.append(
            f"✅ Input: {input_method} → {len(tokens)} tokens extracted"
        )

        # ── Pre-Processor: Row grouping ──
        rows = group_into_rows(tokens)
        result.rows = rows
        result.steps.append(f"✅ Row grouping: {len(rows)} rows formed")

        # ── Pre-Processor: Row classification ──
        classified_rows = classify_rows(rows)
        result.classified_rows = classified_rows
        type_counts = {}
        for rt, _ in classified_rows:
            type_counts[rt] = type_counts.get(rt, 0) + 1
        result.steps.append(f"✅ Row classification: {type_counts}")

        # ── Pre-Processor: Column clustering (optional) ──
        column_info = None
        if use_columns:
            column_info = cluster_columns(rows, classified_rows)
            result.column_info = column_info
            if column_info:
                boundaries, col_names = column_info
                result.steps.append(
                    f"✅ Column detection: {len(boundaries)} columns → "
                    f"{list(col_names.values())}"
                )
            else:
                result.steps.append("⚠️ Column detection: failed, using x-coordinates")
        else:
            result.steps.append("⏭️ Column detection: skipped")

        # ── Pre-Processor: Serialization ──
        serialized_text = serialize(classified_rows, column_info)
        result.serialized_text = serialized_text

        # ── Layer B Pass 1: Rules ──
        rules_fields = {}
        if use_rules:
            rules_fields = run_rules_pass(tokens)
            result.rules_fields = rules_fields
            extracted_list = [k for k in rules_fields if not k.endswith("_valid")]
            result.steps.append(
                f"✅ Rules pass: {', '.join(extracted_list) if extracted_list else 'no fields'}"
            )
        else:
            result.steps.append("⏭️ Rules pass: skipped")

        # ── Layer B Pass 2: LLM ──
        context = rules_fields if inject_context else {}
        llm_result = extract_with_llm(
            serialized_text, context, mode=mode, use_cache=use_cache
        )
        result.llm_raw = llm_result
        result.extracted = llm_result
        result.steps.append("✅ LLM extraction complete (Gemini)")

        # ── Layer C: Confidence ──
        confidences = compute_all_confidences(llm_result, rules_fields, input_method)
        result.confidences = confidences
        invoice_conf = compute_invoice_confidence(confidences)
        result.invoice_confidence = invoice_conf
        result.steps.append(f"✅ Confidence: C_invoice = {invoice_conf:.2f}")

        # ── Layer D: Validation ──
        if mode == "gst":
            invoice, errors = validate_gst_invoice(llm_result)
            result.validated_invoice = invoice
            result.validation_errors = errors
            if not errors:
                # Show math check details
                total = llm_result.get("total_amount")
                taxable = llm_result.get("total_taxable_value", 0) or 0
                cgst = llm_result.get("total_cgst", 0) or 0
                sgst = llm_result.get("total_sgst", 0) or 0
                igst = llm_result.get("total_igst", 0) or 0
                cess = llm_result.get("total_cess", 0) or 0
                expected = taxable + cgst + sgst + igst + cess
                if total:
                    diff = abs(expected - total)
                    result.steps.append(
                        f"✅ Validation passed: |{expected:.2f} - {total:.2f}| = "
                        f"{diff:.2f} ≤ 2.0"
                    )
                else:
                    result.steps.append("✅ Validation passed")
            else:
                result.steps.append(
                    f"⚠️ Validation issues: {'; '.join(errors[:2])}"
                )
        else:
            receipt, errors = validate_sroie_receipt(llm_result)
            result.validated_invoice = receipt
            result.validation_errors = errors
            if not errors:
                result.steps.append("✅ SROIE validation passed")
            else:
                result.steps.append(f"⚠️ SROIE validation: {'; '.join(errors[:2])}")

    except Exception as e:
        logger.exception("Pipeline error")
        result.steps.append(f"❌ Pipeline error: {str(e)}")

    result.processing_time = time.time() - start
    return result
