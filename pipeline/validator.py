"""
Layer D: Pydantic Validators (wrapper module)

Validates extraction results against schemas.
Catches validation errors and returns structured error info
for display in the Streamlit UI.
"""

import logging
from typing import Optional
from pydantic import ValidationError

from schemas.gst_invoice import GSTInvoice
from schemas.sroie_receipt import SROIEReceipt

logger = logging.getLogger(__name__)


def validate_gst_invoice(extracted: dict) -> tuple[Optional[GSTInvoice], list[str]]:
    """
    Validate extracted data against GSTInvoice schema.

    Returns:
        (invoice, errors) where invoice is None if validation fails.
    """
    errors = []
    try:
        invoice = GSTInvoice(**extracted)
        logger.info("GST invoice validation passed")
        return invoice, []
    except ValidationError as e:
        for err in e.errors():
            field = " → ".join(str(loc) for loc in err["loc"])
            msg = err["msg"]
            errors.append(f"{field}: {msg}")
            logger.warning(f"Validation error: {field}: {msg}")
        # Try creating without validators for partial result
        try:
            invoice = GSTInvoice.model_construct(**extracted)
            return invoice, errors
        except Exception:
            return None, errors


def validate_sroie_receipt(extracted: dict) -> tuple[Optional[SROIEReceipt], list[str]]:
    """Validate against SROIEReceipt schema."""
    errors = []
    try:
        receipt = SROIEReceipt(**extracted)
        return receipt, []
    except ValidationError as e:
        for err in e.errors():
            field = " → ".join(str(loc) for loc in err["loc"])
            errors.append(f"{field}: {err['msg']}")
        try:
            receipt = SROIEReceipt.model_construct(**extracted)
            return receipt, errors
        except Exception:
            return None, errors
