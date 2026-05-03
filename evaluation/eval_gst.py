"""
GST Invoice Evaluation Runner

Evaluates the pipeline on synthetic GST invoices with ground truth.
Computes per-field metrics: CER, Token F1, Exact Match.
Also computes line item extraction accuracy and math consistency.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import run_pipeline
from evaluation import compute_cer as cer, compute_token_f1 as token_f1, compute_exact_match as exact_match

logger = logging.getLogger(__name__)

# Fields to evaluate
SCALAR_FIELDS = [
    "seller_name", "seller_address", "seller_gstin",
    "buyer_name", "buyer_address", "buyer_gstin",
    "invoice_number", "invoice_date", "place_of_supply",
    "total_taxable_value", "total_cgst", "total_sgst",
    "total_igst", "total_amount",
]


def evaluate_gst_invoices(
    data_dir: str = "data/gst_invoices",
    output_path: str = "evaluation/gst_results.csv",
    **pipeline_kwargs,
) -> pd.DataFrame:
    """
    Run pipeline on all GST invoice images in data_dir and compare with ground truth.

    Expects pairs: test_invoice_XX.jpg + test_invoice_XX_gt.json
    """
    data_path = Path(data_dir)
    images = sorted(data_path.glob("*.jpg"))

    all_rows = []

    for img_path in images:
        # Find matching ground truth
        stem = img_path.stem  # e.g., "test_invoice_01"
        gt_path = data_path / f"{stem}_gt.json"
        if not gt_path.exists():
            logger.warning(f"No ground truth for {img_path.name}, skipping")
            continue

        with open(gt_path) as f:
            gt = json.load(f)

        print(f"\n{'─'*60}")
        print(f"Processing: {img_path.name}")
        print(f"{'─'*60}")

        start = time.time()
        result = run_pipeline(str(img_path), mode="gst", **pipeline_kwargs)
        elapsed = time.time() - start

        pred = result.extracted
        row = {"file": img_path.name, "time_s": elapsed}

        # Per-field metrics
        for field in SCALAR_FIELDS:
            gt_val = str(gt.get(field, "")) if gt.get(field) is not None else ""
            pred_val = str(pred.get(field, "")) if pred.get(field) is not None else ""

            if gt_val:
                row[f"{field}_cer"] = cer(pred_val, gt_val)
                row[f"{field}_f1"] = token_f1(pred_val, gt_val)
                row[f"{field}_em"] = exact_match(pred_val, gt_val)
            else:
                row[f"{field}_cer"] = None
                row[f"{field}_f1"] = None
                row[f"{field}_em"] = None

        # Line items check
        gt_items = gt.get("items", [])
        pred_items = pred.get("items", [])
        row["gt_items_count"] = len(gt_items)
        row["pred_items_count"] = len(pred_items)
        row["items_count_match"] = len(gt_items) == len(pred_items)

        # Item description accuracy
        if gt_items and pred_items:
            matched = 0
            for gi in gt_items:
                for pi in pred_items:
                    if token_f1(
                        str(pi.get("description", "")),
                        str(gi.get("description", ""))
                    ) > 0.5:
                        matched += 1
                        break
            row["items_desc_recall"] = matched / len(gt_items)
        else:
            row["items_desc_recall"] = 0.0

        # Math consistency
        taxable = pred.get("total_taxable_value") or 0
        cgst = pred.get("total_cgst") or 0
        sgst = pred.get("total_sgst") or 0
        igst = pred.get("total_igst") or 0
        cess = pred.get("total_cess") or 0
        total = pred.get("total_amount") or 0
        if total:
            expected = taxable + cgst + sgst + igst + cess
            row["math_diff"] = abs(expected - total)
            row["math_pass"] = row["math_diff"] <= 2.0
        else:
            row["math_diff"] = None
            row["math_pass"] = False

        # Validation
        row["validation_passed"] = len(result.validation_errors) == 0
        row["confidence"] = result.invoice_confidence

        all_rows.append(row)
        print(f"  → {len([k for k,v in row.items() if k.endswith('_em') and v == 1.0])}"
              f" exact matches, confidence={result.invoice_confidence:.2f}")

    if not all_rows:
        print("No invoices found!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print("GST EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Invoices evaluated: {len(df)}")

    # Field-level averages
    for field in SCALAR_FIELDS:
        cer_col = f"{field}_cer"
        f1_col = f"{field}_f1"
        em_col = f"{field}_em"
        if cer_col in df.columns:
            cer_vals = df[cer_col].dropna()
            f1_vals = df[f1_col].dropna()
            em_vals = df[em_col].dropna()
            if len(cer_vals) > 0:
                print(f"  {field:30s}  CER={cer_vals.mean():.4f}  F1={f1_vals.mean():.4f}  EM={em_vals.mean():.4f}")

    print(f"\n  Items count match: {df['items_count_match'].mean():.2%}")
    print(f"  Items desc recall: {df['items_desc_recall'].mean():.4f}")
    math_pass_vals = df['math_pass'].dropna()
    print(f"  Math consistency:  {math_pass_vals.mean():.2%}")
    print(f"  Avg confidence:    {df['confidence'].mean():.4f}")
    print(f"  Avg time:          {df['time_s'].mean():.2f}s")
    print(f"\nResults saved to {output_path}")

    return df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/gst_invoices")
    parser.add_argument("--output", default="evaluation/gst_results.csv")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    evaluate_gst_invoices(args.data_dir, args.output)
