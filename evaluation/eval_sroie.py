"""
SROIE Dataset Evaluation Runner

Runs the pipeline on SROIE test images and computes metrics:
  - CER (OCR quality)
  - Company F1, Date EM, Total EM
  - Math pass rate (for GST mode)

SROIE ground truth format (keyinfo.txt):
  {"company": "...", "date": "...", "address": "...", "total": "..."}
"""

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import compute_cer, compute_token_f1, compute_exact_match
from pipeline import run_pipeline

logger = logging.getLogger(__name__)


def load_sroie_ground_truth(gt_path: str) -> dict:
    """Load SROIE keyinfo.txt ground truth."""
    with open(gt_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            # Some SROIE files have non-standard format
            lines = f.readlines()
            if len(lines) >= 4:
                return {
                    "company": lines[0].strip(),
                    "date": lines[1].strip(),
                    "address": lines[2].strip(),
                    "total": lines[3].strip(),
                }
            return {}


def evaluate_sroie(
    data_dir: str,
    max_samples: int = 50,
    use_columns: bool = True,
    use_rules: bool = True,
    use_preprocessing: bool = True,
    inject_context: bool = True,
) -> pd.DataFrame:
    """
    Run pipeline on SROIE test set and compute metrics.

    Expected directory structure:
      data_dir/
        img/
          X00001.jpg
          X00002.jpg
          ...
        key/
          X00001.txt (keyinfo JSON)
          X00002.txt
          ...

    Returns:
        DataFrame with per-image metrics
    """
    img_dir = os.path.join(data_dir, "img")
    key_dir = os.path.join(data_dir, "key")

    if not os.path.exists(img_dir):
        # Try flat structure
        img_dir = data_dir
        key_dir = data_dir

    results = []
    image_files = sorted(
        [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    )[:max_samples]

    for img_file in image_files:
        sample_id = Path(img_file).stem
        img_path = os.path.join(img_dir, img_file)
        gt_path = os.path.join(key_dir, f"{sample_id}.txt")

        if not os.path.exists(gt_path):
            logger.warning(f"No ground truth for {sample_id}, skipping")
            continue

        gt = load_sroie_ground_truth(gt_path)
        if not gt:
            continue

        try:
            pipe_result = run_pipeline(
                img_path,
                mode="sroie",
                use_columns=use_columns,
                use_rules=use_rules,
                use_preprocessing=use_preprocessing,
                inject_context=inject_context,
            )

            extracted = pipe_result.extracted or {}

            # Compute metrics (coerce None to empty string)
            company_pred = str(extracted.get("company") or "")
            date_pred = str(extracted.get("date") or "")
            total_pred = str(extracted.get("total") or "")
            address_pred = str(extracted.get("address") or "")

            row = {
                "sample_id": sample_id,
                "company_f1": compute_token_f1(company_pred, gt.get("company", "")),
                "date_em": compute_exact_match(date_pred, gt.get("date", "")),
                "total_em": compute_exact_match(total_pred, gt.get("total", "")),
                "address_f1": compute_token_f1(address_pred, gt.get("address", "")),
                "time_s": pipe_result.processing_time,
            }
            results.append(row)
            logger.info(
                f"{sample_id}: company_f1={row['company_f1']:.2f} "
                f"date_em={row['date_em']:.0f} total_em={row['total_em']:.0f}"
            )

        except Exception as e:
            logger.error(f"Error processing {sample_id}: {e}")
            results.append({
                "sample_id": sample_id,
                "company_f1": 0.0,
                "date_em": 0.0,
                "total_em": 0.0,
                "address_f1": 0.0,
                "time_s": 0.0,
            })

    df = pd.DataFrame(results)
    if not df.empty:
        print("\n=== SROIE Evaluation Results ===")
        print(f"Samples: {len(df)}")
        print(f"Company F1:  {df['company_f1'].mean():.4f}")
        print(f"Date EM:     {df['date_em'].mean():.4f}")
        print(f"Total EM:    {df['total_em'].mean():.4f}")
        print(f"Address F1:  {df['address_f1'].mean():.4f}")
        print(f"Avg time:    {df['time_s'].mean():.2f}s")

    return df


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/sroie_test")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    df = evaluate_sroie(args.data_dir, max_samples=args.max_samples)
    df.to_csv("evaluation/sroie_results.csv", index=False)
    print(f"\nResults saved to evaluation/sroie_results.csv")
