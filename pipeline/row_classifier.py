"""
Row-Type Classifier

Classifies each row into one of:
  - HEADER:    Column title row (contains ≥2 header keywords)
  - LINE_ITEM: Data row with at least one numeric token
  - SUMMARY:   Total/tax rows (contains summary keywords)
  - PRE_TABLE: Rows before the header (seller/buyer info)
  - POST_TABLE: Rows after the last summary (terms, bank details)

Why this matters:
  Without labels, the LLM must infer which rows are line items vs subtotals
  vs headers — a spatial reasoning task it performs inconsistently.
  With explicit labels, the LLM does pure value extraction from
  pre-classified rows.

CRITICAL FIX (from review):
  Previous version used "OTHER (ignore these rows)" — but rows before
  the header contain seller name and address! Renamed to PRE_TABLE/POST_TABLE
  with explicit instructions to extract info from them.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ingestion import Token

logger = logging.getLogger(__name__)

HEADER_KEYWORDS = {
    "description",
    "item",
    "particulars",
    "product",
    "goods",
    "qty",
    "quantity",
    "nos",
    "units",
    "pcs",
    "rate",
    "price",
    "mrp",
    "unit price",
    "amount",
    "value",
    "taxable",
    "hsn",
    "sac",
    "code",
    "discount",
    "sl",
    "sr",
    "no",
    "s.no",
}

SUMMARY_KEYWORDS = {
    "subtotal",
    "sub total",
    "sub-total",
    "total",
    "grand total",
    "net total",
    "net amount",
    "cgst",
    "sgst",
    "igst",
    "gst",
    "tax",
    "cess",
    "round off",
    "round-off",
    "rounding",
    "balance",
    "amount due",
    "amount payable",
    "invoice total",
    "payable",
}


def _classify_single_row(row_tokens: list["Token"]) -> str:
    """
    Classify a single row based on its content.

    Returns one of: "HEADER", "LINE_ITEM", "SUMMARY", "CONTEXT"
    (CONTEXT is later refined to PRE_TABLE or POST_TABLE based on position)
    """
    text = " ".join(t.text.lower() for t in row_tokens)

    # HEADER: contains ≥2 header keywords (it's the column title row)
    header_matches = sum(1 for kw in HEADER_KEYWORDS if kw in text)
    if header_matches >= 2:
        return "HEADER"

    # SUMMARY: contains any summary keyword
    if any(kw in text for kw in SUMMARY_KEYWORDS):
        return "SUMMARY"

    # LINE_ITEM: has at least one pure numeric token (price, qty, amount)
    has_number = any(
        t.text.replace(".", "", 1).replace(",", "").isdigit()
        for t in row_tokens
    )
    if has_number:
        return "LINE_ITEM"

    # CONTEXT: everything else (seller info, buyer info, footer text, etc.)
    return "CONTEXT"


def classify_rows(
    rows: list[list["Token"]],
) -> list[tuple[str, list["Token"]]]:
    """
    Classify all rows and refine CONTEXT into PRE_TABLE/POST_TABLE.

    PRE_TABLE: CONTEXT rows before the first HEADER row
      → Contains seller name, buyer name, addresses, GSTINs
      → LLM should extract seller/buyer info from these rows

    POST_TABLE: CONTEXT rows after the last SUMMARY row
      → Contains terms, bank details, signatures
      → LLM may find place of supply here

    Returns:
        List of (row_type, row_tokens) tuples
    """
    # First pass: classify each row
    classified = [(_classify_single_row(row), row) for row in rows]

    # Find header and last summary positions
    header_idx = None
    last_summary_idx = None

    for i, (row_type, _) in enumerate(classified):
        if row_type == "HEADER" and header_idx is None:
            header_idx = i
        if row_type == "SUMMARY":
            last_summary_idx = i

    # Second pass: refine CONTEXT → PRE_TABLE or POST_TABLE
    result = []
    for i, (row_type, row_tokens) in enumerate(classified):
        if row_type == "CONTEXT":
            if header_idx is not None and i < header_idx:
                row_type = "PRE_TABLE"
            elif last_summary_idx is not None and i > last_summary_idx:
                row_type = "POST_TABLE"
            else:
                # Between header and summary but not a line item or summary
                # Could be a multi-line description continuation
                row_type = "PRE_TABLE"  # safe default
        result.append((row_type, row_tokens))

    # Log summary
    type_counts = {}
    for rt, _ in result:
        type_counts[rt] = type_counts.get(rt, 0) + 1
    logger.info(f"Row classifier: {type_counts}")

    return result
