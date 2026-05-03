"""
Serializer: Converts 2D spatial structure into LLM-ready prompt text.

Two serialization modes:
  1. Named columns (when column clustering succeeds):
     Row 3 [LINE_ITEM]: Description="Mouse" | Qty="2" | Rate="400" | Amount="800"

  2. X-coordinate fallback (when column clustering fails):
     Row 3 [LINE_ITEM]: [x=10] "Mouse" [x=165] "2" [x=225] "400" [x=295] "800"

Row-type labels (PRE_TABLE, HEADER, LINE_ITEM, SUMMARY, POST_TABLE) tell the LLM
exactly which rows to extract line items from and which contain seller/buyer info.
"""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .ingestion import Token

from .column_clusterer import assign_tokens_to_columns

logger = logging.getLogger(__name__)


def serialize_with_columns(
    classified_rows: list[tuple[str, list["Token"]]],
    boundaries: list[float],
    col_names: dict[int, str],
) -> str:
    """
    Serialize rows using named columns.

    Produces:
      PRE_TABLE (extract seller/buyer info from these rows):
        Row 0 [PRE_TABLE]: "ABC Electronics Pvt Ltd"
        Row 1 [PRE_TABLE]: "GSTIN: 29ABCDE1234F1Z5"

      HEADER ROW:
        Row 2: Description | Qty | Rate | Amount

      LINE ITEMS (extract each as a LineItem):
        Row 3 [LINE_ITEM]: Description="Mouse" | Qty="2" | ...

      SUMMARY (extract tax and total values):
        Row 5 [SUMMARY]: "CGST @ 9%" → "94.50"
    """
    sections = {
        "PRE_TABLE": [],
        "HEADER": [],
        "LINE_ITEM": [],
        "SUMMARY": [],
        "POST_TABLE": [],
    }

    for row_idx, (row_type, row_tokens) in enumerate(classified_rows):
        if row_type == "HEADER":
            # Serialize header as column names
            col_assignments = assign_tokens_to_columns(row_tokens, boundaries)
            header_parts = []
            for token, col_idx in zip(row_tokens, col_assignments):
                name = col_names.get(col_idx, f"col_{col_idx}")
                header_parts.append(name)
            line = f"  Row {row_idx}: {' | '.join(header_parts)}"
            sections["HEADER"].append(line)

        elif row_type == "LINE_ITEM":
            # Serialize with named column assignments
            col_assignments = assign_tokens_to_columns(row_tokens, boundaries)
            parts = []
            for token, col_idx in zip(row_tokens, col_assignments):
                name = col_names.get(col_idx, f"col_{col_idx}")
                parts.append(f'{name}="{token.text}"')
            line = f"  Row {row_idx} [LINE_ITEM]: {' | '.join(parts)}"
            sections["LINE_ITEM"].append(line)

        elif row_type == "SUMMARY":
            text = " ".join(f'"{t.text}"' for t in row_tokens)
            line = f"  Row {row_idx} [SUMMARY]: {text}"
            sections["SUMMARY"].append(line)

        elif row_type in ("PRE_TABLE", "POST_TABLE"):
            text = " ".join(f'"{t.text}"' for t in row_tokens)
            line = f"  Row {row_idx} [{row_type}]: {text}"
            sections[row_type].append(line)

    return _build_prompt_text(sections)


def serialize_with_coordinates(
    classified_rows: list[tuple[str, list["Token"]]],
) -> str:
    """
    Fallback serialization using x-coordinates instead of column names.

    Produces:
      Row 3 [LINE_ITEM]: [x=10] "Mouse" [x=165] "2" [x=225] "400" [x=295] "800"
    """
    sections = {
        "PRE_TABLE": [],
        "HEADER": [],
        "LINE_ITEM": [],
        "SUMMARY": [],
        "POST_TABLE": [],
    }

    for row_idx, (row_type, row_tokens) in enumerate(classified_rows):
        parts = " ".join(f'[x={t.x_min}] "{t.text}"' for t in row_tokens)
        line = f"  Row {row_idx} [{row_type}]: {parts}"

        if row_type in sections:
            sections[row_type].append(line)
        else:
            sections.setdefault("PRE_TABLE", []).append(line)

    return _build_prompt_text(sections)


def _build_prompt_text(sections: dict[str, list[str]]) -> str:
    """Build the final structured prompt text from classified sections."""
    parts = ["DOCUMENT STRUCTURE:\n"]

    if sections.get("PRE_TABLE"):
        parts.append(
            "PRE_TABLE (extract seller name, buyer name, addresses, GSTINs from these rows):"
        )
        parts.extend(sections["PRE_TABLE"])
        parts.append("")

    if sections.get("HEADER"):
        parts.append("HEADER ROW:")
        parts.extend(sections["HEADER"])
        parts.append("")

    if sections.get("LINE_ITEM"):
        parts.append("LINE ITEMS (extract each as a LineItem object):")
        parts.extend(sections["LINE_ITEM"])
        parts.append("")

    if sections.get("SUMMARY"):
        parts.append("SUMMARY SECTION (extract tax and total values):")
        parts.extend(sections["SUMMARY"])
        parts.append("")

    if sections.get("POST_TABLE"):
        parts.append(
            "POST_TABLE (may contain place of supply or additional info):"
        )
        parts.extend(sections["POST_TABLE"])
        parts.append("")

    return "\n".join(parts)


def serialize(
    classified_rows: list[tuple[str, list["Token"]]],
    column_info: Optional[tuple[list[float], dict[int, str]]] = None,
) -> str:
    """
    Main serialization entry point.

    Args:
        classified_rows: list of (row_type, row_tokens) from row_classifier
        column_info: (boundaries, col_names) from column_clusterer, or None

    Returns:
        Structured text ready for LLM prompt injection.
    """
    if column_info is not None:
        boundaries, col_names = column_info
        text = serialize_with_columns(classified_rows, boundaries, col_names)
        logger.info("Serialized with named columns")
    else:
        text = serialize_with_coordinates(classified_rows)
        logger.info("Serialized with x-coordinate fallback")

    return text
