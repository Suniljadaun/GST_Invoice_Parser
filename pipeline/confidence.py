"""
Layer C: Confidence Scoring (Display Only)

C_field = 0.7 * mean(c_i) + 0.3 * min(c_i)

Risk-sensitive estimator: C ≈ E[c] - λ·risk, λ=0.3
A single low-confidence OCR token pulls the field score down.

Confidence categories are DETERMINISTIC by field name (not arbitrary):
  - Well-anchored fields (total, seller_name): 0.85
  - Loosely-anchored fields (addresses, items): 0.65
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Fields that have clear spatial anchors on invoices → higher LLM confidence
WELL_ANCHORED_FIELDS = {
    "total_amount", "seller_name", "invoice_date", "invoice_number",
    "total_cgst", "total_sgst", "total_igst", "total_taxable_value",
}

# Fields with ambiguous boundaries → lower LLM confidence
LOOSELY_ANCHORED_FIELDS = {
    "seller_address", "buyer_address", "buyer_name",
    "place_of_supply", "items", "total_cess",
}


def compute_field_confidence(
    field_name: str,
    value,
    extraction_source: str,
    ocr_confidences: Optional[list[float]] = None,
) -> float:
    """
    Compute confidence score for a single extracted field.

    Args:
        field_name: name of the field
        value: extracted value (None → 0.0 confidence)
        extraction_source: "rules", "text_pdf", "gemini"
        ocr_confidences: list of OCR token confidences that contributed

    Returns:
        Confidence score in [0.0, 1.0]
    """
    if value is None:
        return 0.0

    # Rules-extracted fields: deterministically correct
    if extraction_source == "rules":
        return 1.0

    # Text-PDF path: no OCR noise
    if extraction_source == "text_pdf":
        return 1.0

    # Gemini-extracted: confidence depends on field type
    if extraction_source == "gemini":
        if field_name in WELL_ANCHORED_FIELDS:
            return 0.85
        elif field_name in LOOSELY_ANCHORED_FIELDS:
            return 0.65
        else:
            return 0.75  # default for unknown fields

    # OCR-based with token confidences
    if ocr_confidences and len(ocr_confidences) > 0:
        mean_c = sum(ocr_confidences) / len(ocr_confidences)
        min_c = min(ocr_confidences)
        return 0.7 * mean_c + 0.3 * min_c

    return 0.75  # fallback


def compute_all_confidences(
    extracted: dict,
    rules_fields: dict,
    input_method: str,
) -> dict[str, float]:
    """
    Compute confidence scores for all extracted fields.

    Returns:
        Dict mapping field_name → confidence score
    """
    confidences = {}

    for field_name, value in extracted.items():
        if field_name in ("items",):
            # Items get a blanket confidence based on source
            confidences[field_name] = 0.65 if input_method != "text_pdf" else 0.90
            continue

        if field_name in rules_fields:
            source = "rules"
        elif input_method == "text_pdf":
            source = "text_pdf"
        else:
            source = "gemini"

        confidences[field_name] = compute_field_confidence(
            field_name, value, source
        )

    return confidences


def get_confidence_color(score: float) -> str:
    """Map confidence score to UI color indicator."""
    if score >= 0.90:
        return "🟢"
    elif score >= 0.75:
        return "🟡"
    else:
        return "🔴"


def compute_invoice_confidence(
    confidences: dict[str, float],
) -> float:
    """
    Compute overall invoice-level confidence score.
    Weighted average with field importance weights.
    """
    FIELD_WEIGHTS = {
        "seller_gstin": 3.0,
        "invoice_number": 2.0,
        "invoice_date": 2.0,
        "total_amount": 3.0,
        "items": 2.0,
        "seller_name": 1.0,
    }

    weighted_sum = 0.0
    weight_total = 0.0

    for field, weight in FIELD_WEIGHTS.items():
        if field in confidences:
            weighted_sum += confidences[field] * weight
            weight_total += weight

    if weight_total == 0:
        return 0.0

    return weighted_sum / weight_total
