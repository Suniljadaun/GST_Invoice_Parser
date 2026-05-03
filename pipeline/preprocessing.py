"""
Layer 0: Image Preprocessing

Two-step preprocessing for scanned documents:
  1. Deskew — corrects rotation using minAreaRect on dark pixels
  2. CLAHE — Contrast Limited Adaptive Histogram Equalization

Why CLAHE instead of adaptive thresholding:
  PaddleOCR's recognition model (CRNN/SVTR) was trained on grayscale images.
  Adaptive thresholding outputs binary (0 or 255), destroying the grayscale
  gradients the recognizer uses to distinguish similar characters ('5' vs '6').
  CLAHE enhances local contrast while preserving grayscale information.
"""

import cv2
import numpy as np


def deskew_image(gray: np.ndarray) -> np.ndarray:
    """
    Correct slight rotation from scanning.
    Uses minAreaRect on dark pixel coordinates to estimate skew angle.
    Only corrects if skew > 0.5 degrees (avoids unnecessary interpolation).
    """
    # Find dark pixel coordinates (text regions)
    coords = np.column_stack(np.where(gray < 128))

    if len(coords) < 100:
        # Not enough dark pixels to estimate angle reliably
        return gray

    angle = cv2.minAreaRect(coords)[-1]

    # minAreaRect returns angle in [-90, 0). Normalize to [-45, 45).
    if angle < -45:
        angle += 90

    # Only correct if skew is significant
    if abs(angle) <= 0.5:
        return gray

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        gray,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return deskewed


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Enhances local contrast for faded thermal prints (common in SROIE)
    without destroying grayscale gradients needed by PaddleOCR.

    Parameters:
      clipLimit=2.0   — limits contrast amplification to avoid noise boost
      tileGridSize=(8,8) — processes 8×8 tiles independently
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline: BGR → grayscale → deskew → CLAHE.

    Args:
        image: BGR image (from cv2.imread or pdf2image)

    Returns:
        Enhanced grayscale image ready for PaddleOCR
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Step 1: Deskew
    gray = deskew_image(gray)

    # Step 2: CLAHE (contrast enhancement, preserves grayscale)
    enhanced = apply_clahe(gray)

    return enhanced
