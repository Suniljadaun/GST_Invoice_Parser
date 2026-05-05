#!/usr/bin/env bash
# Download and prepare SROIE dataset for evaluation.
# Requires: pip install kaggle  (and ~/.kaggle/kaggle.json configured)
#
# SROIE: ICDAR 2019 Robust Reading Challenge on Scanned Receipts
# 1,000 receipt images with OCR + key-info JSON ground truth.
# Source: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2
#
# After download the dataset extracts to:
#   data/archive/SROIE2019/test/img/       (347 .jpg files)
#   data/archive/SROIE2019/test/entities/  (347 .txt JSON ground truth files)
#
# eval_sroie.py and ablation.py use --data-dir data/archive/SROIE2019/test
# which maps img/ -> img/ and key/ -> entities/ automatically.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Downloading SROIE dataset from Kaggle ==="
kaggle datasets download urbikn/sroie-datasetv2 -p data/archive --unzip

echo "=== Verifying extracted structure ==="
# Dataset extracts to data/archive/SROIE2019/test/{img,entities}/
IMG_DIR="data/archive/SROIE2019/test/img"
KEY_DIR="data/archive/SROIE2019/test/entities"

if [ ! -d "$IMG_DIR" ]; then
    echo "ERROR: Expected $IMG_DIR not found after extraction."
    echo "Check data/archive/ for the actual structure and update paths."
    exit 1
fi

echo "=== Done ==="
echo "Images:  $(ls "$IMG_DIR"/*.jpg 2>/dev/null | wc -l)"
echo "Keys:    $(ls "$KEY_DIR"/*.txt 2>/dev/null | wc -l)"
echo ""
echo "Run evaluation with:"
echo "  python evaluation/eval_sroie.py --data-dir data/archive/SROIE2019/test"
echo "  python evaluation/ablation.py   --data-dir data/archive/SROIE2019/test"
