"""
Evaluation Metrics: CER, Token F1, Exact Match

CER (Character Error Rate) = edit_distance(pred, gt) / len(gt)
Token F1: whitespace-split, precision/recall on token sets
Exact Match: binary, 1 if pred == gt else 0
"""

import editdistance


def compute_cer(predicted: str, ground_truth: str) -> float:
    """Character Error Rate = edit_distance / len(ground_truth)."""
    if not ground_truth:
        return 0.0 if not predicted else 1.0
    return editdistance.eval(predicted, ground_truth) / len(ground_truth)


def compute_token_f1(predicted: str, ground_truth: str) -> float:
    """Token-level F1 score (whitespace-split)."""
    pred_tokens = set(predicted.lower().split())
    gt_tokens = set(ground_truth.lower().split())
    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    tp = len(pred_tokens & gt_tokens)
    precision = tp / len(pred_tokens) if pred_tokens else 0.0
    recall = tp / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(predicted: str, ground_truth: str) -> float:
    """Binary exact match (case-insensitive, stripped)."""
    return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0
