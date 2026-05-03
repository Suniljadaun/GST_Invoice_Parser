"""
Layer B Pass 1: Deterministic Rules Engine

Three-level GSTIN validation:
  Level 1 — Regex: catches wrong formats
  Level 2 — Structural: state code ∈ {01–38}, char-14 = 'Z'
  Level 3 — Checksum: modified Luhn algorithm on base-36 character set

Seller vs Buyer GSTIN disambiguation:
  Sort all matched GSTINs by y_min position.
  Topmost → seller (letterhead region).
  Second → buyer (Bill To section).

Other fields: date (3 formats), invoice number (keyword-anchored),
phone (word-bounded), PIN code (word-bounded with context).
"""

import re
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .ingestion import Token

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# GSTIN validation (3 levels)
# ──────────────────────────────────────────────────────────

GSTIN_REGEX = re.compile(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]")

VALID_STATE_CODES = {
    "01", "02", "03", "04", "05", "06", "07", "08", "09",
    "10", "11", "12", "13", "14", "15", "16", "17", "18",
    "19", "20", "21", "22", "23", "24", "25", "26", "27",
    "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38",
}


def validate_gstin_structure(gstin: str) -> tuple[bool, str]:
    """
    Level 2: Structural validation.
    Checks state code is in valid set and character 14 is 'Z'.
    """
    state_code = gstin[:2]
    char_14 = gstin[13]

    if state_code not in VALID_STATE_CODES:
        return False, f"Invalid state code: {state_code}"
    if char_14 != "Z":
        return False, f"Character 14 must be 'Z', got '{char_14}'"
    return True, "OK"


def validate_gstin_checksum(gstin: str) -> tuple[bool, str]:
    """
    Level 3: Checksum validation using modified Luhn algorithm (base-36).

    Character set: 0–9 then A–Z (base 36)
    Position weight: odd positions (0-indexed) ×1, even positions ×2
    Reduction: digit//36 + digit%36 (single base-36 digit)
    Check digit: (36 - total%36) % 36

    This catches single-character OCR misreads that are invisible to
    regex and structural checks.
    """
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    if len(gstin) != 15:
        return False, f"GSTIN must be 15 characters, got {len(gstin)}"

    total = 0
    for i, char in enumerate(gstin[:14]):
        char_upper = char.upper()
        if char_upper not in CHARS:
            return False, f"Invalid character '{char}' at position {i + 1}"

        digit = CHARS.index(char_upper)

        # Odd positions (0-indexed) are multiplied by 2
        if i % 2 != 0:
            digit *= 2

        # Reduce to single digit in base-36
        total += digit // 36 + digit % 36

    remainder = total % 36
    expected_check = CHARS[(36 - remainder) % 36]
    actual_check = gstin[14].upper()

    if actual_check != expected_check:
        return False, f"Checksum failed: expected '{expected_check}', got '{actual_check}'"

    return True, "Valid"


def extract_and_assign_gstins(tokens: list["Token"]) -> dict:
    """
    Find all GSTINs and assign seller vs buyer based on y-position.

    Standard GST invoice layout:
      Seller GSTIN → top section (letterhead / company info block)
      Buyer GSTIN → billing/shipping section (lower on page)
    """
    matches = []
    for token in tokens:
        text_upper = token.text.upper().replace(" ", "")
        m = GSTIN_REGEX.search(text_upper)
        if m:
            gstin = m.group(0)

            # Must pass structural validation at minimum
            valid_struct, struct_msg = validate_gstin_structure(gstin)
            valid_check, check_msg = validate_gstin_checksum(gstin)

            if valid_struct:
                matches.append({
                    "gstin": gstin,
                    "y_min": token.bbox[1],
                    "structural_valid": True,
                    "checksum_valid": valid_check,
                    "checksum_msg": check_msg,
                })

    # Sort by vertical position (topmost first)
    matches.sort(key=lambda x: x["y_min"])

    result = {
        "seller_gstin": None,
        "buyer_gstin": None,
        "seller_gstin_checksum_valid": None,
        "buyer_gstin_checksum_valid": None,
    }

    if len(matches) >= 1:
        result["seller_gstin"] = matches[0]["gstin"]
        result["seller_gstin_checksum_valid"] = matches[0]["checksum_valid"]
    if len(matches) >= 2:
        result["buyer_gstin"] = matches[1]["gstin"]
        result["buyer_gstin_checksum_valid"] = matches[1]["checksum_valid"]

    if matches:
        logger.info(
            f"GSTINs found: {len(matches)} "
            f"(seller={result['seller_gstin']}, buyer={result['buyer_gstin']})"
        )

    return result


# ──────────────────────────────────────────────────────────
# Date extraction (3 format patterns)
# ──────────────────────────────────────────────────────────

DATE_PATTERNS = [
    # DD-MM-YYYY or DD/MM/YYYY
    re.compile(r"\b(\d{2}[-/]\d{2}[-/]\d{4})\b"),
    # YYYY-MM-DD
    re.compile(r"\b(\d{4}[-/]\d{2}[-/]\d{2})\b"),
    # DD Month YYYY (e.g., "12 April 2024")
    re.compile(
        r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"(?:uary|ruary|ch|il|e|y|ust|tember|ober|ember)?\s+\d{4})\b",
        re.IGNORECASE,
    ),
]


def extract_date(full_text: str) -> Optional[str]:
    """Try all date patterns, return first match."""
    for pattern in DATE_PATTERNS:
        m = pattern.search(full_text)
        if m:
            return m.group(1)
    return None


# ──────────────────────────────────────────────────────────
# Invoice number extraction (keyword-anchored)
# ──────────────────────────────────────────────────────────

INV_KEYWORDS = {"invoice", "inv", "bill", "receipt", "voucher", "memo"}
# "no" removed — too broad, matches "No" in any context

# Words that should never be an invoice number
INV_REJECT = {"terms", "payment", "within", "days", "bank", "account", "note",
              "total", "amount", "tax", "gst", "cgst", "sgst", "igst"}


def extract_invoice_number(
    tokens: list["Token"],
    full_text: str,
) -> Optional[str]:
    """
    Extract invoice number using keyword-anchored approach.

    Strategy 0: Handle colon-joined tokens (e.g., "Invoice No:TW/2024/001").
    Strategy 1: Find "Invoice No" label, take the next token with digits.
    Strategy 2: Fallback to regex pattern.
    """
    # Strategy 0: colon-joined tokens like "Invoice No:TW/2024/001"
    for token in tokens:
        text = token.text.strip()
        if ":" in text:
            parts = text.split(":", 1)
            label = parts[0].lower().strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            # Check if label contains an invoice keyword + "no"/"number"/"num"
            has_inv_kw = any(kw in label for kw in INV_KEYWORDS)
            has_no = any(n in label for n in ("no", "num", "number", "#"))
            if has_inv_kw and has_no and value and any(c.isdigit() for c in value):
                return value

    # Strategy 1: keyword-anchored (look at next tokens)
    for i, token in enumerate(tokens):
        text_lower = token.text.lower().strip(":. ")
        # Skip generic "TAX INVOICE" header — it's a title, not a number label
        if "tax" in text_lower and "invoice" in text_lower:
            continue
        if any(kw in text_lower for kw in INV_KEYWORDS):
            # Look at next 1–4 tokens for the actual number
            for j in range(i + 1, min(i + 5, len(tokens))):
                candidate = tokens[j].text.strip(":. ")
                # Skip empty, too short
                if len(candidate) < 2:
                    continue
                # Reject if ANY word in the candidate is in the reject list
                candidate_words = set(candidate.lower().replace(":", " ").split())
                if candidate_words & INV_REJECT:
                    continue
                # Must contain at least one digit
                if not any(c.isdigit() for c in candidate):
                    continue
                # Skip if it matches a date pattern
                if re.match(r"\d{2}[-/]\d{2}[-/]\d{4}", candidate):
                    continue
                # Skip if it looks like a GSTIN
                if len(candidate) == 15 and GSTIN_REGEX.match(candidate.upper()):
                    continue
                return candidate

    # Strategy 2: fallback regex
    patterns = [
        re.compile(r"[A-Z]{2,4}[-/][0-9]{3,}"),
        re.compile(r"#\s*(\d{4,})"),
        re.compile(r"[A-Z]{2,4}\d{4,}"),
    ]
    for pattern in patterns:
        m = pattern.search(full_text)
        if m:
            return m.group(0)

    return None


# ──────────────────────────────────────────────────────────
# Phone and PIN (word-bounded)
# ──────────────────────────────────────────────────────────

PHONE_PATTERN = re.compile(r"\b([6-9]\d{9})\b")
PIN_PATTERN = re.compile(r"\b([1-9]\d{5})\b")


def extract_phone(full_text: str) -> Optional[str]:
    """Extract Indian phone number with word boundaries."""
    m = PHONE_PATTERN.search(full_text)
    return m.group(1) if m else None


def extract_pin_code(full_text: str) -> Optional[str]:
    """
    Extract PIN code with context validation.
    PIN codes appear near address-like context, not in isolation.
    """
    PIN_CONTEXT_KEYWORDS = {"pin", "code", "zip", "india", "road", "street",
                            "nagar", "colony", "sector", "phase", "block"}

    for m in PIN_PATTERN.finditer(full_text):
        pin = m.group(1)
        # Don't match if it's part of a phone number (10 digits)
        start = m.start()
        end = m.end()
        # Check surrounding characters aren't digits
        if start > 0 and full_text[start - 1].isdigit():
            continue
        if end < len(full_text) and full_text[end].isdigit():
            continue

        # Check for address context nearby
        context_start = max(0, start - 80)
        context = full_text[context_start:end].lower()
        if any(kw in context for kw in PIN_CONTEXT_KEYWORDS):
            return pin

    # Fallback: return last 6-digit number (PINs are often at end of address)
    matches = list(PIN_PATTERN.finditer(full_text))
    if matches:
        return matches[-1].group(1)

    return None


# ──────────────────────────────────────────────────────────
# Main rules pass
# ──────────────────────────────────────────────────────────

def run_rules_pass(tokens: list["Token"]) -> dict:
    """
    Extract all deterministically extractable fields using regex rules.

    Fields extracted:
      - seller_gstin, buyer_gstin (with checksum validation)
      - seller_pan (derived from seller GSTIN chars 3–12)
      - invoice_date (3 format patterns)
      - invoice_number (keyword-anchored + fallback regex)
      - phone, pin_code (word-bounded with context)

    Returns:
        dict of extracted field name → value
    """
    # Reconstruct full text from tokens
    full_text = " ".join(t.text for t in tokens)

    extracted = {}

    # GSTIN (seller + buyer, spatial disambiguation)
    gstin_info = extract_and_assign_gstins(tokens)
    if gstin_info["seller_gstin"]:
        extracted["seller_gstin"] = gstin_info["seller_gstin"]
        extracted["seller_gstin_checksum_valid"] = gstin_info["seller_gstin_checksum_valid"]
    if gstin_info["buyer_gstin"]:
        extracted["buyer_gstin"] = gstin_info["buyer_gstin"]
        extracted["buyer_gstin_checksum_valid"] = gstin_info["buyer_gstin_checksum_valid"]

    # PAN: derived from seller GSTIN characters 3–12
    if extracted.get("seller_gstin"):
        extracted["seller_pan"] = extracted["seller_gstin"][2:12]

    # Date
    date = extract_date(full_text)
    if date:
        extracted["invoice_date"] = date

    # Invoice number
    inv_num = extract_invoice_number(tokens, full_text)
    if inv_num:
        extracted["invoice_number"] = inv_num

    # Phone
    phone = extract_phone(full_text)
    if phone:
        extracted["phone"] = phone

    # PIN code
    pin = extract_pin_code(full_text)
    if pin:
        extracted["pin_code"] = pin

    logger.info(f"Rules pass: extracted {len(extracted)} fields: {list(extracted.keys())}")
    return extracted
