"""
GSTInvoice and LineItem Pydantic schemas.

Used for:
  - Gemini structured output (response_schema=GSTInvoice)
  - Pydantic validation (CGST·IGST=0, math consistency)
  - Streamlit data editor display
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class LineItem(BaseModel):
    """A single line item row in the invoice table."""
    model_config = ConfigDict(coerce_numbers_to_str=True, str_strip_whitespace=True)

    description: Optional[str] = Field("", description="Item description / product name")
    hsn_sac_code: Optional[str] = Field(
        None, description="HSN code (goods) or SAC code (services). null if not visible."
    )
    quantity: Optional[float] = Field(
        None, description="Quantity. null if not visible — never 0.0"
    )
    unit: Optional[str] = Field(None, description="Unit of measurement (pcs, kg, hrs, etc.)")
    rate: Optional[float] = Field(
        None, description="Unit price / rate. null if not visible — never 0.0"
    )
    discount: Optional[float] = Field(None, description="Discount amount on this line item")
    taxable_value: Optional[float] = Field(None, description="Row subtotal / taxable amount")
    cgst_rate: Optional[float] = Field(None, description="CGST rate % for this item")
    sgst_rate: Optional[float] = Field(None, description="SGST rate % for this item")
    igst_rate: Optional[float] = Field(None, description="IGST rate % for this item")


class GSTInvoice(BaseModel):
    """
    Complete GST invoice schema with two validators:
      1. Tax type consistency: CGST · IGST = 0
      2. Math consistency: |Total - Components| ≤ ε (ε=2.0)
    """

    model_config = ConfigDict(coerce_numbers_to_str=True, str_strip_whitespace=True)

    # --- Seller ---
    seller_name: Optional[str] = Field(None, description="Seller / company name")
    seller_address: Optional[str] = Field(None, description="Seller address")
    seller_gstin: Optional[str] = Field(None, description="Seller GSTIN (15 chars)")
    seller_pan: Optional[str] = Field(None, description="Seller PAN (derived from GSTIN chars 3–12)")

    # --- Buyer ---
    buyer_name: Optional[str] = Field(None, description="Buyer / bill-to name")
    buyer_address: Optional[str] = Field(None, description="Buyer address")
    buyer_gstin: Optional[str] = Field(None, description="Buyer GSTIN (15 chars)")

    # --- Invoice details ---
    invoice_number: Optional[str] = Field(None, description="Invoice / bill number")
    invoice_date: Optional[str] = Field(None, description="Invoice date as printed on document")
    place_of_supply: Optional[str] = Field(
        None, description="Place of supply (determines CGST+SGST vs IGST)"
    )

    # --- Line items ---
    items: List[LineItem] = Field(default_factory=list, description="Extracted line items")

    # --- Totals ---
    total_taxable_value: Optional[float] = Field(None, description="Sum of all taxable values")
    total_cgst: Optional[float] = Field(None, description="Total CGST amount. None, never 0.0")
    total_sgst: Optional[float] = Field(None, description="Total SGST amount. None, never 0.0")
    total_igst: Optional[float] = Field(None, description="Total IGST amount (inter-state)")
    total_cess: Optional[float] = Field(None, description="Total cess amount")
    total_amount: Optional[float] = Field(None, description="Grand total / amount payable")

    # ──────────────────────────────────────────────
    # Validator 1 — Invoice type consistency
    # An invoice cannot have both CGST and IGST.
    # Intra-state supply → CGST + SGST
    # Inter-state supply → IGST only
    # ──────────────────────────────────────────────
    @model_validator(mode="after")
    def check_tax_type_consistency(self) -> "GSTInvoice":
        has_cgst = self.total_cgst is not None and self.total_cgst > 0
        has_igst = self.total_igst is not None and self.total_igst > 0
        if has_cgst and has_igst:
            raise ValueError(
                "Structural error: Invoice has both CGST and IGST. "
                "Intra-state supply uses CGST+SGST only. "
                "Inter-state supply uses IGST only. Cannot have both."
            )
        return self

    # ──────────────────────────────────────────────
    # Validator 2 — Math consistency with ε = 2.0
    #
    # Why ε = 2.0:
    #   Tax rounding:  9% of ₹100.50 = ₹9.045 → printed ₹9.05
    #   Multi-line accumulation ≈ ±₹0.50
    #   OCR digit transposition ≤ ₹1.00
    #   ε = 2.0 covers rounding+OCR noise, rejects structural failures
    #
    # CRITICAL: use `is not None` never `or 0.0`
    #   0.0 is falsy → `x or 0.0` treats legitimate zero as missing
    # ──────────────────────────────────────────────
    @model_validator(mode="after")
    def check_math_consistency(self) -> "GSTInvoice":
        taxable = self.total_taxable_value if self.total_taxable_value is not None else 0.0
        cgst = self.total_cgst if self.total_cgst is not None else 0.0
        sgst = self.total_sgst if self.total_sgst is not None else 0.0
        igst = self.total_igst if self.total_igst is not None else 0.0
        cess = self.total_cess if self.total_cess is not None else 0.0
        total = self.total_amount if self.total_amount is not None else 0.0

        expected = taxable + cgst + sgst + igst + cess
        diff = abs(expected - total)

        EPSILON = 2.0

        if diff > EPSILON and total > 0 and expected > 0:
            raise ValueError(
                f"Math validation failed. "
                f"Components sum = {expected:.2f} "
                f"(taxable={taxable:.2f} + cgst={cgst:.2f} + sgst={sgst:.2f} "
                f"+ igst={igst:.2f} + cess={cess:.2f}), "
                f"parsed total = {total:.2f}, "
                f"diff = {diff:.2f} > ε={EPSILON}"
            )
        return self
