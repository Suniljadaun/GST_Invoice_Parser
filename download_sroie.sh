#!/usr/bin/env bash
# Download and prepare SROIE dataset for evaluation.
# Requires: pip install kaggle  (and ~/.kaggle/kaggle.json configured)
#
# SROIE: ICDAR 2019 Robust Reading Challenge on Scanned Receipts
# 1,000 receipt images with OCR + key-info JSON ground truth.
# Source: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Downloading SROIE dataset from Kaggle ==="
kaggle datasets download urbikn/sroie-datasetv2 -p data/sroie_raw --unzip

echo "=== Copying test images and key files ==="
# The dataset structure varies; find the test images and keys
find data/sroie_raw -name "*.jpg" -path "*/test/*" | head -20 | while read f; do
    cp "$f" data/sroie_test/img/
done

find data/sroie_raw -name "*.txt" -path "*/test/*key*" -o -name "*.json" -path "*/test/*key*" | head -20 | while read f; do
    cp "$f" data/sroie_test/key/
done

# If no test split found, just take last 20 images
if [ "$(ls -A data/sroie_test/img/ 2>/dev/null | wc -l)" -eq 0 ]; then
    echo "No test split found, taking last 20 images..."
    find data/sroie_raw -name "*.jpg" | sort | tail -20 | while read f; do
        cp "$f" data/sroie_test/img/
    done
fi

echo "=== Done ==="
echo "Images: $(ls data/sroie_test/img/*.jpg 2>/dev/null | wc -l)"
echo "Keys:   $(ls data/sroie_test/key/* 2>/dev/null | wc -l)"
