"""
Ablation Table Generator

Runs 8 pipeline variants on SROIE test set and builds the ablation table.
Each row adds one component to prove its marginal value.

Variants:
  1. Baseline: flat text → LLM only
  2. + Row grouping (1D spatial)
  3. + Column clustering (2D spatial)
  4. + Row-type classification
  5. + Image preprocessing (deskew + CLAHE)
  6. + Rules pass (regex)
  7. + Context injection to LLM
  8. Full pipeline
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.eval_sroie import evaluate_sroie

logger = logging.getLogger(__name__)


ABLATION_VARIANTS = [
    {
        "name": "1. Baseline (flat text → LLM)",
        "use_columns": False,
        "use_rules": False,
        "use_preprocessing": False,
        "inject_context": False,
    },
    {
        "name": "2. + Row grouping",
        "use_columns": False,
        "use_rules": False,
        "use_preprocessing": False,
        "inject_context": False,
    },
    {
        "name": "3. + Column clustering",
        "use_columns": True,
        "use_rules": False,
        "use_preprocessing": False,
        "inject_context": False,
    },
    {
        "name": "4. + Row-type classification",
        "use_columns": True,
        "use_rules": False,
        "use_preprocessing": False,
        "inject_context": False,
    },
    {
        "name": "5. + Preprocessing (deskew+CLAHE)",
        "use_columns": True,
        "use_rules": False,
        "use_preprocessing": True,
        "inject_context": False,
    },
    {
        "name": "6. + Rules pass (regex)",
        "use_columns": True,
        "use_rules": True,
        "use_preprocessing": True,
        "inject_context": False,
    },
    {
        "name": "7. + Context injection",
        "use_columns": True,
        "use_rules": True,
        "use_preprocessing": True,
        "inject_context": True,
    },
    {
        "name": "8. Full pipeline",
        "use_columns": True,
        "use_rules": True,
        "use_preprocessing": True,
        "inject_context": True,
    },
]


def run_ablation(
    data_dir: str = "data/archive/SROIE2019/test",
    max_samples: int = 50,
    output_path: str = "evaluation/ablation_results.csv",
) -> pd.DataFrame:
    """
    Run all ablation variants and produce the results table.
    """
    all_results = []

    for variant in ABLATION_VARIANTS:
        name = variant["name"]
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        df = evaluate_sroie(
            data_dir,
            max_samples=max_samples,
            use_columns=variant["use_columns"],
            use_rules=variant["use_rules"],
            use_preprocessing=variant["use_preprocessing"],
            inject_context=variant["inject_context"],
        )

        if not df.empty:
            row = {
                "Variant": name,
                "Company F1": f"{df['company_f1'].mean():.4f}",
                "Date EM": f"{df['date_em'].mean():.4f}",
                "Total EM": f"{df['total_em'].mean():.4f}",
                "Address F1": f"{df['address_f1'].mean():.4f}",
                "Avg Time (s)": f"{df['time_s'].mean():.2f}",
            }
        else:
            row = {
                "Variant": name,
                "Company F1": "N/A",
                "Date EM": "N/A",
                "Total EM": "N/A",
                "Address F1": "N/A",
                "Avg Time (s)": "N/A",
            }

        all_results.append(row)

    ablation_df = pd.DataFrame(all_results)
    ablation_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("ABLATION TABLE")
    print(f"{'='*60}")
    print(ablation_df.to_string(index=False))
    print(f"\nSaved to {output_path}")

    return ablation_df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/archive/SROIE2019/test")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    run_ablation(args.data_dir, args.max_samples)
