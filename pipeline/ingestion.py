"""
Layer A: Dual-Path Ingestion

Path 1 — Text PDF (pypdf):
  If the PDF contains extractable text with real words, skip OCR entirely.
  Faster, more accurate, no bounding box noise. c_i = 1.0 for all tokens.

Path 2 — Image / Scanned PDF (PaddleOCR):
  Uses Differentiable Binarization (DB) network.
  B = 1 / (1 + exp(-k(P - T))), k ≈ 50
  Adaptively computes bounding boxes for skewed receipt text.

Post-OCR: Token deduplication via IoU-based Non-Maximum Suppression.
Multi-page support: y-coordinates offset by page_num × PAGE_HEIGHT.
"""

import re
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pypdf
from PIL import Image

from .preprocessing import preprocess_for_ocr

logger = logging.getLogger(__name__)

# Standard A4 at 150 DPI ≈ 1754 × 2481 px. Use 2500 as a safe page height.
PAGE_HEIGHT = 2500


# ──────────────────────────────────────────────────────────
# Token type
# ──────────────────────────────────────────────────────────
class Token:
    """A single OCR token with bounding box, text, and confidence."""

    __slots__ = ("bbox", "text", "confidence", "page")

    def __init__(
        self,
        bbox: list[int],
        text: str,
        confidence: float = 1.0,
        page: int = 0,
    ):
        # bbox = [x_min, y_min, x_max, y_max]
        self.bbox = bbox
        self.text = text
        self.confidence = confidence
        self.page = page

    @property
    def y_center(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def x_min(self) -> int:
        return self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "text": self.text,
            "confidence": self.confidence,
            "page": self.page,
        }

    def __repr__(self) -> str:
        return f"Token('{self.text}', bbox={self.bbox}, c={self.confidence:.2f})"


# ──────────────────────────────────────────────────────────
# Text PDF detection (robust)
# ──────────────────────────────────────────────────────────
def is_usable_text_pdf(pdf_path: str) -> bool:
    """
    Returns True only if pypdf extracts meaningful text.
    Checks alphanumeric count AND presence of real English words.
    Prevents garbage-encoding PDFs from bypassing OCR.
    """
    try:
        reader = pypdf.PdfReader(pdf_path)
        full_text = "".join(page.extract_text() or "" for page in reader.pages)

        alnum_count = sum(1 for c in full_text if c.isalnum())
        has_real_words = bool(re.search(r"[a-zA-Z]{3,}", full_text))

        return alnum_count > 50 and has_real_words
    except Exception:
        return False


# ──────────────────────────────────────────────────────────
# Path 1: Text PDF extraction
# ──────────────────────────────────────────────────────────
def extract_from_text_pdf(pdf_path: str) -> list[Token]:
    """
    Extract tokens from a native text PDF using pypdf.
    All tokens get confidence = 1.0 (deterministic extraction).
    Tokens are split by whitespace; bounding boxes are estimated
    from character positions since pypdf doesn't provide spatial info.
    """
    reader = pypdf.PdfReader(pdf_path)
    tokens = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        lines = text.split("\n")

        for line_idx, line in enumerate(lines):
            words = line.split()
            x_offset = 10  # estimated starting x position

            for word in words:
                if not word.strip():
                    continue

                # Estimate bbox from line/word position
                # These are approximate — text PDFs don't have real bboxes
                y_min = 50 + line_idx * 25 + page_num * PAGE_HEIGHT
                y_max = y_min + 20
                x_min = x_offset
                x_max = x_min + len(word) * 8  # ~8px per character

                tokens.append(
                    Token(
                        bbox=[x_min, y_min, x_max, y_max],
                        text=word,
                        confidence=1.0,
                        page=page_num,
                    )
                )
                x_offset = x_max + 10  # gap between words

    logger.info(f"Text PDF: extracted {len(tokens)} tokens from {len(reader.pages)} pages")
    return tokens


# ──────────────────────────────────────────────────────────
# Path 2: PaddleOCR extraction
# ──────────────────────────────────────────────────────────
def _get_ocr_engine():
    """Lazy-load PaddleOCR — handles both old API (<=2.7) and new API (>=2.8)."""
    from paddleocr import PaddleOCR
    import inspect

    sig = inspect.signature(PaddleOCR.__init__).parameters

    # Old API: PaddleOCR(use_angle_cls, lang, show_log)
    # New API: PaddleOCR() — no constructor args, uses predict() not ocr()
    if "lang" in sig:
        kwargs = {"use_angle_cls": True, "lang": "en"}
        if "show_log" in sig:
            kwargs["show_log"] = False
        return PaddleOCR(**kwargs)
    else:
        return PaddleOCR()


def _run_ocr(ocr_engine, image: np.ndarray) -> list:
    """
    Version-aware OCR call.
    Old API: ocr_engine.ocr(img, cls=True)  → list of [[bbox, (text, conf)], ...]
    New API: ocr_engine.predict(img)         → list of result objects
    """
    import inspect

    # Try old API first
    if hasattr(ocr_engine, "ocr"):
        try:
            sig = inspect.signature(ocr_engine.ocr)
            if "cls" in sig.parameters:
                return ocr_engine.ocr(image, cls=True)
            else:
                return ocr_engine.ocr(image)
        except TypeError:
            pass

    # New API: predict() returns a list of OCRResult objects
    results = ocr_engine.predict(image)
    if not results:
        return [[]]

    # Convert new API result format → old format [[bbox_points, (text, conf)], ...]
    converted = []
    for res in results:
        texts = getattr(res, "rec_texts", []) or []
        scores = getattr(res, "rec_scores", []) or []
        polys = getattr(res, "det_polys", None) or getattr(res, "boxes", []) or []

        page_lines = []
        for i, text in enumerate(texts):
            conf = scores[i] if i < len(scores) else 1.0
            if i < len(polys):
                poly = polys[i]
                if hasattr(poly, "tolist"):
                    poly = poly.tolist()
                if isinstance(poly[0], (list, tuple)):
                    bbox_points = poly
                else:
                    x1, y1, x2, y2 = poly[0], poly[1], poly[2], poly[3]
                    bbox_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            else:
                bbox_points = [[0, 0], [100, 0], [100, 20], [0, 20]]
            page_lines.append([bbox_points, (text, conf)])
        converted.append(page_lines)

    return converted


def extract_from_image(
    image: np.ndarray,
    ocr_engine,
    page_num: int = 0,
    apply_preprocessing: bool = True,
) -> list[Token]:
    """
    Run PaddleOCR on a single image (one page).

    Args:
        image: BGR numpy array
        ocr_engine: PaddleOCR instance
        page_num: page number for y-offset in multi-page docs
        apply_preprocessing: whether to apply deskew + CLAHE
    """
    if apply_preprocessing:
        processed = preprocess_for_ocr(image)
        # PaddleOCR expects BGR or grayscale. Convert back to 3-channel.
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    else:
        processed = image

    results = _run_ocr(ocr_engine, processed)

    tokens = []
    if results and results[0]:
        for line in results[0]:
            bbox_points, (text, confidence) = line

            # Convert polygon points to [x_min, y_min, x_max, y_max]
            x_min = int(min(pt[0] for pt in bbox_points))
            y_min = int(min(pt[1] for pt in bbox_points)) + page_num * PAGE_HEIGHT
            x_max = int(max(pt[0] for pt in bbox_points))
            y_max = int(max(pt[1] for pt in bbox_points)) + page_num * PAGE_HEIGHT

            tokens.append(
                Token(
                    bbox=[x_min, y_min, x_max, y_max],
                    text=text.strip(),
                    confidence=confidence,
                    page=page_num,
                )
            )

    return tokens


def extract_from_scanned_pdf(
    pdf_path: str,
    ocr_engine,
    apply_preprocessing: bool = True,
) -> list[Token]:
    """
    Convert each PDF page to image, run PaddleOCR with y-offset per page.
    Multi-page support: all pages unified into a single coordinate space.
    """
    from pdf2image import convert_from_path

    pages = convert_from_path(pdf_path, dpi=150)
    all_tokens = []

    for page_num, page_image in enumerate(pages):
        page_array = np.array(page_image)  # PIL → numpy (RGB)
        page_bgr = cv2.cvtColor(page_array, cv2.COLOR_RGB2BGR)

        page_tokens = extract_from_image(
            page_bgr, ocr_engine, page_num=page_num,
            apply_preprocessing=apply_preprocessing,
        )
        all_tokens.extend(page_tokens)

    logger.info(
        f"Scanned PDF: extracted {len(all_tokens)} tokens from {len(pages)} pages"
    )
    return all_tokens


# ──────────────────────────────────────────────────────────
# Token deduplication via IoU / NMS
# ──────────────────────────────────────────────────────────
def compute_iou(bbox_a: list[int], bbox_b: list[int]) -> float:
    """Intersection over Union for two [x1, y1, x2, y2] bounding boxes."""
    x_left = max(bbox_a[0], bbox_b[0])
    y_top = max(bbox_a[1], bbox_b[1])
    x_right = min(bbox_a[2], bbox_b[2])
    y_bottom = min(bbox_a[3], bbox_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def deduplicate_tokens(tokens: list[Token], iou_threshold: float = 0.5) -> list[Token]:
    """
    Non-Maximum Suppression: keep highest-confidence token when two overlap.
    PaddleOCR occasionally produces duplicate detections for bold/shadowed text.
    """
    sorted_tokens = sorted(tokens, key=lambda t: -t.confidence)
    kept: list[Token] = []

    for token in sorted_tokens:
        is_duplicate = any(
            compute_iou(token.bbox, kept_token.bbox) > iou_threshold
            for kept_token in kept
        )
        if not is_duplicate:
            kept.append(token)

    if len(kept) < len(tokens):
        logger.info(
            f"Deduplication: removed {len(tokens) - len(kept)} duplicate tokens"
        )

    return kept


# ──────────────────────────────────────────────────────────
# Main ingestion entry point
# ──────────────────────────────────────────────────────────
def ingest(
    file_path: str,
    apply_preprocessing: bool = True,
) -> tuple[list[Token], str]:
    """
    Main ingestion function. Decides between text-PDF and OCR paths.

    Args:
        file_path: path to PDF or image file
        apply_preprocessing: whether to apply deskew + CLAHE before OCR

    Returns:
        (tokens, input_method) where input_method is 'text_pdf' or 'paddleocr'
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf" and is_usable_text_pdf(file_path):
        # Path 1: Text PDF
        tokens = extract_from_text_pdf(file_path)
        return tokens, "text_pdf"

    # Path 2: OCR
    ocr_engine = _get_ocr_engine()

    if suffix == ".pdf":
        tokens = extract_from_scanned_pdf(
            file_path, ocr_engine, apply_preprocessing=apply_preprocessing
        )
    elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Failed to read image: {file_path}")
        tokens = extract_from_image(
            image, ocr_engine, page_num=0,
            apply_preprocessing=apply_preprocessing,
        )
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # Post-OCR deduplication
    tokens = deduplicate_tokens(tokens)

    return tokens, "paddleocr"
