"""
Layer B Pass 2: Gemini LLM Extraction

Uses Gemini structured output (response_mime_type="application/json").
Features: context injection, MD5 disk cache, rate limiting, post-processing.
"""

import hashlib, json, logging, os, re, time
from typing import Optional

logger = logging.getLogger(__name__)
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "llm_cache")


def clean_numeric(value) -> Optional[float]:
    """Deterministic currency string -> float. Done in post-processing, NOT by LLM."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[₹$€£\s,]", "", str(value))
    cleaned = cleaned.replace("Rs.", "").replace("Rs", "").replace("INR", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def build_extraction_prompt(serialized_text: str, rules_fields: dict) -> str:
    """Build LLM prompt with context-injected rules fields."""
    ctx = []
    if rules_fields:
        ctx.append("PRE-EXTRACTED BY RULES ENGINE (do not re-extract or modify these):")
        display = {"seller_gstin": "Seller GSTIN", "buyer_gstin": "Buyer GSTIN",
                    "invoice_number": "Invoice Number", "invoice_date": "Invoice Date",
                    "seller_pan": "Seller PAN"}
        for key, label in display.items():
            if key in rules_fields:
                ck = f"{key}_checksum_valid"
                if ck in rules_fields:
                    st = "valid" if rules_fields[ck] else "invalid"
                    ctx.append(f"  {label}: {rules_fields[key]}  [checksum: {st}]")
                else:
                    ctx.append(f"  {label}: {rules_fields[key]}")
        ctx.append("")
    context_block = "\n".join(ctx)
    return f"""{context_block}
{serialized_text}

INSTRUCTIONS:
1. Extract seller name, seller address, buyer name, buyer address from PRE_TABLE rows.
2. Extract all LINE_ITEM rows into items list with description, quantity, rate, taxable_value.
3. Extract SUMMARY values into total_cgst, total_sgst, total_igst, total_cess, total_amount.
4. Extract total_taxable_value (sum before taxes).
5. Check POST_TABLE rows for place_of_supply.
6. If a field is not present, set it to null.
7. Do not invent or calculate values - extract only what is printed.
8. For numeric fields, return the number as-is from the document.
""".strip()


def build_sroie_prompt(serialized_text: str) -> str:
    return f"""Extract these fields from this receipt:
- company: store/company name
- date: receipt date
- address: store address
- total: total amount (as string, exactly as printed)

{serialized_text}

Return these 4 fields only."""


def _get_gemini_model():
    import google.generativeai as genai
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY env var. "
            "Free key: https://aistudio.google.com/apikey"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def call_gemini_cached(prompt: str, use_cache: bool = True) -> dict:
    """Gemini call with MD5 cache + rate limiting + exponential backoff."""
    import google.generativeai as genai
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = hashlib.md5(prompt.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            logger.info(f"LLM cache hit: {cache_key[:8]}")
            return json.load(f)

    model = _get_gemini_model()
    # Use JSON mode only — schema validation done by Pydantic in Layer D
    config = genai.GenerationConfig(response_mime_type="application/json")

    for attempt in range(3):
        try:
            resp = model.generate_content(contents=prompt, generation_config=config)
            result = json.loads(resp.text)
            with open(cache_path, "w") as f:
                json.dump(result, f, indent=2)
            time.sleep(4.1)  # 60/15 = 4s between calls
            logger.info(f"LLM extraction complete (attempt {attempt + 1})")
            return result
        except Exception as e:
            es = str(e).lower()
            is_rate_limit = "429" in str(e) or "resource_exhausted" in es or "quota" in es
            if is_rate_limit:
                wait = 15 * (2 ** attempt)  # 15s, 30s, 60s
                logger.warning(f"Rate limited. Waiting {wait}s (attempt {attempt + 1}/3)...")
                time.sleep(wait)
            else:
                # Non-rate-limit errors: fail immediately, don't waste retries
                logger.error(f"Gemini API error: {e}")
                raise
    raise RuntimeError("Gemini call failed after 3 rate-limit retries")


NUMERIC_FIELDS = {
    "quantity", "rate", "discount", "taxable_value",
    "cgst_rate", "sgst_rate", "igst_rate",
    "total_taxable_value", "total_cgst", "total_sgst",
    "total_igst", "total_cess", "total_amount",
}


def postprocess_extracted(raw: dict) -> dict:
    """Post-process: strip currency symbols from numeric fields."""
    result = dict(raw)
    for field in NUMERIC_FIELDS:
        if field in result:
            result[field] = clean_numeric(result[field])
    if "items" in result and isinstance(result["items"], list):
        for item in result["items"]:
            if isinstance(item, dict):
                for field in NUMERIC_FIELDS:
                    if field in item:
                        item[field] = clean_numeric(item[field])
    return result


def extract_with_llm(serialized_text: str, rules_fields: dict,
                     mode: str = "gst", use_cache: bool = True) -> dict:
    """Full LLM extraction: build prompt -> call Gemini -> post-process -> merge."""
    if mode == "sroie":
        prompt = build_sroie_prompt(serialized_text)
        return call_gemini_cached(prompt, use_cache=use_cache)

    prompt = build_extraction_prompt(serialized_text, rules_fields)
    raw = call_gemini_cached(prompt, use_cache=use_cache)

    raw = postprocess_extracted(raw)
    # Rules fields take precedence (higher confidence)
    for field in ["seller_gstin", "buyer_gstin", "seller_pan", "invoice_date", "invoice_number"]:
        if field in rules_fields and rules_fields[field] is not None:
            raw[field] = rules_fields[field]
    return raw
