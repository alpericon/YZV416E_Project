"""
J&F evaluation for Video Object Segmentation.

Implements:
- J (Jaccard / IoU) score: region similarity
- F (boundary F-score): contour accuracy
- Mean J&F

Compatible with DAVIS 2017 evaluation conventions.
All functions operate on binary numpy masks (uint8, 0/1).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


# ---------------------------------------------------------------------------
# J score (Jaccard / IoU)
# ---------------------------------------------------------------------------

def jaccard(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Jaccard index (IoU) between two binary masks.
    Returns value in [0, 1]. Returns 1.0 when both masks are empty.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if not gt.any() and not pred.any():
        return 1.0
    if not gt.any() or not pred.any():
        return 0.0

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection) / float(union)


# ---------------------------------------------------------------------------
# F score (boundary F-score)
# ---------------------------------------------------------------------------

def _mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """
    Extract boundary pixels from a binary mask using morphological erosion.
    dilation_ratio controls the boundary thickness relative to image diagonal.
    """
    h, w = mask.shape
    diag = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * diag)))

    mask = mask.astype(np.uint8)
    # Boundary = mask XOR eroded_mask
    eroded = binary_erosion(mask, iterations=dilation).astype(np.uint8)
    boundary = mask - eroded
    return boundary.astype(bool)


def f_score(pred: np.ndarray, gt: np.ndarray, dilation_ratio: float = 0.02) -> float:
    """
    Compute boundary F-score between predicted and ground-truth masks.
    Returns value in [0, 1]. Returns 1.0 when both masks are empty.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if not gt.any() and not pred.any():
        return 1.0
    if not gt.any() or not pred.any():
        return 0.0

    pred_boundary = _mask_to_boundary(pred.astype(np.uint8), dilation_ratio)
    gt_boundary = _mask_to_boundary(gt.astype(np.uint8), dilation_ratio)

    # Dilate boundaries for tolerance
    h, w = gt.shape
    diag = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * diag)))

    pred_dil = binary_dilation(pred_boundary, iterations=dilation)
    gt_dil = binary_dilation(gt_boundary, iterations=dilation)

    precision = np.logical_and(pred_boundary, gt_dil).sum() / (pred_boundary.sum() + 1e-8)
    recall = np.logical_and(gt_boundary, pred_dil).sum() / (gt_boundary.sum() + 1e-8)

    if precision + recall < 1e-8:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


# ---------------------------------------------------------------------------
# Per-sequence and dataset-level evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    compute_f: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a single sequence.

    Args:
        pred_masks: list of (H, W) binary uint8 masks (one per frame)
        gt_masks:   list of (H, W) binary uint8 masks (one per frame)
        compute_f:  whether to compute F score (slower, requires boundary extraction)

    Returns:
        dict with keys 'J', 'F', 'JF' (all averaged over frames)
    """
    assert len(pred_masks) == len(gt_masks), \
        f"Pred/GT length mismatch: {len(pred_masks)} vs {len(gt_masks)}"

    j_scores = [jaccard(p, g) for p, g in zip(pred_masks, gt_masks)]
    result = {"J": float(np.mean(j_scores))}

    if compute_f:
        f_scores = [f_score(p, g) for p, g in zip(pred_masks, gt_masks)]
        result["F"] = float(np.mean(f_scores))
        result["JF"] = (result["J"] + result["F"]) / 2.0
    else:
        result["JF"] = result["J"]

    return result


def evaluate_dataset(
    predictions: Dict[str, Dict[int, List[np.ndarray]]],
    ground_truth: Dict[str, Dict[int, List[np.ndarray]]],
    compute_f: bool = True,
) -> Dict[str, object]:
    """
    Evaluate over all sequences and objects.

    Args:
        predictions: {seq_name: {obj_id: [binary_masks]}}
        ground_truth: {seq_name: {obj_id: [binary_masks]}}

    Returns:
        {
            'per_sequence': {seq_name: {obj_id: {'J', 'F', 'JF'}}},
            'mean_J': float,
            'mean_F': float,
            'mean_JF': float,   ← the headline metric
        }
    """
    per_seq = {}
    all_j, all_f = [], []

    for seq_name, obj_preds in predictions.items():
        per_seq[seq_name] = {}
        gt_objs = ground_truth.get(seq_name, {})

        for obj_id, pred_masks in obj_preds.items():
            gt_masks = gt_objs.get(obj_id, [])
            if not gt_masks:
                continue

            result = evaluate_sequence(pred_masks, gt_masks, compute_f=compute_f)
            per_seq[seq_name][obj_id] = result
            all_j.append(result["J"])
            if compute_f:
                all_f.append(result["F"])

    mean_j = float(np.mean(all_j)) if all_j else 0.0
    mean_f = float(np.mean(all_f)) if all_f else 0.0
    mean_jf = (mean_j + mean_f) / 2.0 if all_f else mean_j

    return {
        "per_sequence": per_seq,
        "mean_J": mean_j,
        "mean_F": mean_f,
        "mean_JF": mean_jf,
    }


def print_results(results: Dict) -> None:
    """Pretty-print evaluation results table."""
    print("\n" + "=" * 60)
    print(f"{'Sequence':<25} {'J':>8} {'F':>8} {'J&F':>8}")
    print("-" * 60)

    seq_jf = []
    for seq_name, obj_dict in results["per_sequence"].items():
        j_vals = [v["J"] for v in obj_dict.values()]
        f_vals = [v.get("F", v["J"]) for v in obj_dict.values()]
        jf_vals = [v["JF"] for v in obj_dict.values()]
        mean_j = np.mean(j_vals)
        mean_f = np.mean(f_vals)
        mean_jf = np.mean(jf_vals)
        seq_jf.append(mean_jf)
        print(f"{seq_name:<25} {mean_j:>8.4f} {mean_f:>8.4f} {mean_jf:>8.4f}")

    print("=" * 60)
    print(
        f"{'MEAN':<25} "
        f"{results['mean_J']:>8.4f} "
        f"{results['mean_F']:>8.4f} "
        f"{results['mean_JF']:>8.4f}"
    )
    print("=" * 60)


def save_results_csv(results: Dict, output_path: str) -> None:
    """Save per-sequence results to CSV."""
    import csv, os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "obj_id", "J", "F", "JF"])
        for seq_name, obj_dict in results["per_sequence"].items():
            for obj_id, metrics in obj_dict.items():
                writer.writerow([
                    seq_name, obj_id,
                    f"{metrics['J']:.6f}",
                    f"{metrics.get('F', metrics['J']):.6f}",
                    f"{metrics['JF']:.6f}",
                ])
        writer.writerow([])
        writer.writerow(["MEAN", "", f"{results['mean_J']:.6f}",
                          f"{results['mean_F']:.6f}", f"{results['mean_JF']:.6f}"])
    print(f"Results saved to {output_path}")
