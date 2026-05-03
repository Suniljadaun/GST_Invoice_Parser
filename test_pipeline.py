"""Quick end-to-end test of the pipeline."""
import sys, os, json, logging

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from pipeline import run_pipeline

print("=" * 60)
print("TESTING PIPELINE ON SYNTHETIC GST INVOICE")
print("=" * 60)

result = run_pipeline("data/gst_invoices/test_invoice_01.jpg", mode="gst")

print("\n--- Pipeline Steps ---")
for step in result.steps:
    print(f"  {step}")

print(f"\nProcessing time: {result.processing_time:.2f}s")
print(f"Input method: {result.input_method}")
print(f"Tokens: {len(result.tokens)}")
print(f"Rows: {len(result.rows)}")
print(f"Invoice confidence: {result.invoice_confidence:.2f}")

print("\n--- Extracted Data ---")
for k, v in result.extracted.items():
    if k != "items":
        print(f"  {k}: {v}")

if "items" in result.extracted and result.extracted["items"]:
    print(f"\n--- Line Items ({len(result.extracted['items'])}) ---")
    for i, item in enumerate(result.extracted["items"]):
        print(f"  Item {i+1}: {item}")

if result.validation_errors:
    print("\n--- Validation Errors ---")
    for err in result.validation_errors:
        print(f"  ⚠️  {err}")
else:
    print("\n✅ All validations passed!")
