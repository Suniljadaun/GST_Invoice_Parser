"""
Row Grouper: Anchored Y-Center Algorithm

Groups OCR tokens into horizontal rows based on vertical proximity.

Key design decisions:
  1. Dynamic threshold: Δy = median(token heights) / 2
     → Scale-invariant across DPI and font sizes.
  2. Anchored comparison: each token is compared to the FIRST token's
     y-center in the row, not a running mean.

Why anchored (not running mean):
  Running mean drifts: μ_{k+1} = (k·μ_k + y_{k+1}) / (k+1)
  For y = [100, 102, 104, 106] with Δy = 4:
    μ → 103, so |106 - 103| = 3 ≤ 4 → false merge of two lines.
  Anchored approach prevents drift entirely.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ingestion import Token

logger = logging.getLogger(__name__)


def compute_delta_y(tokens: list["Token"]) -> float:
    """
    Dynamic row-grouping threshold = half the median token height.
    Scale-invariant — adapts to DPI and font size automatically.
    """
    heights = [t.height for t in tokens if t.height > 0]
    if not heights:
        return 10.0  # fallback

    heights.sort()
    mid = len(heights) // 2
    if len(heights) % 2 == 0:
        median_height = (heights[mid - 1] + heights[mid]) / 2.0
    else:
        median_height = heights[mid]

    return max(median_height / 2.0, 3.0)  # minimum 3px to avoid over-splitting


def group_into_rows(tokens: list["Token"]) -> list[list["Token"]]:
    """
    Group tokens into horizontal rows using anchored y-center comparison.

    Algorithm:
      1. Sort tokens by y_center ascending
      2. anchor_y = y_center of first token (set once per row, never updated)
      3. For each token:
           if |y_center - anchor_y| ≤ Δy → same row
           else → start new row, reset anchor
      4. Within each row: sort by x_min (left → right)

    Returns:
        List of rows, where each row is a list of Tokens sorted left-to-right.
    """
    if not tokens:
        return []

    delta_y = compute_delta_y(tokens)
    logger.info(f"Row grouper: Δy = {delta_y:.1f}px, {len(tokens)} tokens")

    # Sort by y_center
    sorted_tokens = sorted(tokens, key=lambda t: t.y_center)

    rows: list[list["Token"]] = []
    current_row: list["Token"] = [sorted_tokens[0]]
    anchor_y = sorted_tokens[0].y_center  # set once, never updated

    for token in sorted_tokens[1:]:
        if abs(token.y_center - anchor_y) <= delta_y:
            current_row.append(token)
        else:
            # Save current row sorted left-to-right
            current_row.sort(key=lambda t: t.x_min)
            rows.append(current_row)
            # Start new row with new anchor
            current_row = [token]
            anchor_y = token.y_center

    # Don't forget the last row
    current_row.sort(key=lambda t: t.x_min)
    rows.append(current_row)

    logger.info(f"Row grouper: formed {len(rows)} rows")
    return rows
