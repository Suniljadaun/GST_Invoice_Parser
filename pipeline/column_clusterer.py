"""
Column Clusterer: X-Axis Gap Detection

Groups tokens into logical columns based on their x_min positions.
This is an OPTIONAL, ISOLATED module — the system works without it.

Method: Gap detection on sorted x_min values.
  Column boundaries = gaps significantly larger than the mean gap.
  Threshold = mean(gaps) + 1.5 * std(gaps)

Fallback: If column detection fails or produces > 8 columns
  (noise, not real columns), discard and use x-coordinate serialization.

Why optional:
  SROIE receipts have ragged alignment — right-justified numbers don't
  have consistent x_min positions. Column clustering helps on well-formatted
  invoices but can hurt on messy ones. Making it optional lets us quantify
  its value in the ablation table.
"""

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .ingestion import Token

logger = logging.getLogger(__name__)

MAX_COLUMNS = 8  # If more columns detected, column clustering is noise


def detect_column_boundaries(tokens: list["Token"]) -> Optional[list[float]]:
    """
    Detect column boundaries from token x_min positions using gap analysis.

    Returns:
        List of column boundary x-positions, or None if detection fails.
    """
    if len(tokens) < 5:
        return None

    x_mins = sorted(set(int(t.x_min) for t in tokens))

    if len(x_mins) < 3:
        return None

    # Compute pairwise gaps
    gaps = [x_mins[i + 1] - x_mins[i] for i in range(len(x_mins) - 1)]

    if not gaps:
        return None

    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)

    # Column boundaries = positions where gap exceeds threshold
    threshold = mean_gap + 1.5 * std_gap

    boundaries = [float(x_mins[0])]  # first column starts at leftmost x
    for i, gap in enumerate(gaps):
        if gap > threshold:
            boundaries.append(float(x_mins[i + 1]))

    # Sanity check: too many columns = noise
    if len(boundaries) > MAX_COLUMNS:
        logger.warning(
            f"Column detection found {len(boundaries)} columns (> {MAX_COLUMNS}). "
            f"Discarding — likely noise."
        )
        return None

    if len(boundaries) < 2:
        logger.info("Column detection: only 1 column found, skipping.")
        return None

    logger.info(f"Column detection: {len(boundaries)} columns at x = {boundaries}")
    return boundaries


def assign_tokens_to_columns(
    tokens: list["Token"],
    boundaries: list[float],
) -> list[int]:
    """
    Assign each token to the nearest column boundary.

    Returns:
        List of column indices (same length as tokens).
    """
    col_assignments = []
    for token in tokens:
        x = float(token.x_min)
        # Find nearest boundary
        min_dist = float("inf")
        best_col = 0
        for col_idx, boundary in enumerate(boundaries):
            dist = abs(x - boundary)
            if dist < min_dist:
                min_dist = dist
                best_col = col_idx
        col_assignments.append(best_col)
    return col_assignments


def detect_column_names(
    header_row: list["Token"],
    boundaries: list[float],
) -> dict[int, str]:
    """
    Label columns by matching header row tokens to known field keywords.

    Maps column indices to semantic names:
      {"description", "item", "particulars"} → "Description"
      {"qty", "quantity", "nos", "units"}    → "Qty"
      {"rate", "price", "mrp"}               → "Rate"
      {"amount", "total", "value"}           → "Amount"
      {"hsn", "sac"}                         → "HSN/SAC"
      {"discount"}                           → "Discount"
    """
    COLUMN_KEYWORD_MAP = {
        "Description": {"description", "item", "particulars", "product", "goods"},
        "Qty": {"qty", "quantity", "nos", "units", "pcs"},
        "Rate": {"rate", "price", "mrp", "unit price"},
        "Amount": {"amount", "value", "taxable"},
        "HSN/SAC": {"hsn", "sac", "code"},
        "Discount": {"discount", "disc"},
    }

    col_assignments = assign_tokens_to_columns(header_row, boundaries)
    col_names: dict[int, str] = {}

    for token, col_idx in zip(header_row, col_assignments):
        text_lower = token.text.lower().strip()
        for col_name, keywords in COLUMN_KEYWORD_MAP.items():
            if text_lower in keywords or any(kw in text_lower for kw in keywords):
                col_names[col_idx] = col_name
                break

    # Fill unnamed columns with positional names
    for i in range(len(boundaries)):
        if i not in col_names:
            col_names[i] = f"col_{i}"

    logger.info(f"Column names: {col_names}")
    return col_names


def cluster_columns(
    rows: list[list["Token"]],
    classified_rows: list[tuple[str, list["Token"]]],
) -> Optional[tuple[list[float], dict[int, str]]]:
    """
    Full column clustering pipeline.

    Returns:
        (boundaries, column_names) or None if clustering fails.
    """
    # Collect all tokens across all rows for boundary detection
    all_tokens = [t for row in rows for t in row]
    boundaries = detect_column_boundaries(all_tokens)

    if boundaries is None:
        return None

    # Find header row for column naming
    header_row = None
    for row_type, row_tokens in classified_rows:
        if row_type == "HEADER":
            header_row = row_tokens
            break

    if header_row:
        col_names = detect_column_names(header_row, boundaries)
    else:
        # No header found — use positional names
        col_names = {i: f"col_{i}" for i in range(len(boundaries))}
        logger.info("No header row found — using positional column names")

    return boundaries, col_names
