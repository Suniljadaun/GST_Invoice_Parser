"""
SROIE Receipt schema — used only for evaluation on the SROIE dataset.

SROIE ground truth has exactly 4 fields per receipt:
  company, date, address, total

This schema maps our pipeline output to those 4 fields for metric computation.
"""

from pydantic import BaseModel, Field


class SROIEReceipt(BaseModel):
    """Minimal schema matching SROIE keyinfo.txt ground-truth format."""

    company: str = Field(..., description="Company / store name (maps to SROIE 'company')")
    date: str = Field(..., description="Receipt date (maps to SROIE 'date')")
    address: str = Field(..., description="Store address (maps to SROIE 'address')")
    total: str = Field(..., description="Total amount as string (maps to SROIE 'total')")
