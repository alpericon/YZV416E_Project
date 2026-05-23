"""
Visualization utilities for SAM 2 Video Object Segmentation results.

Produces:
- Mask overlays on video frames
- Side-by-side prediction vs. ground-truth comparisons
- J&F score plots (per-sequence bar chart, training curves)
- Error analysis heatmaps (TP/FP/FN regions)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# Distinct colors for up to 10 objects (RGB 0-255)
OBJECT_COLORS = [
    (255, 0, 0),    # red
    (0, 255, 0),    # green
    (0, 0, 255),    # blue
    (255, 165, 0),  # orange
    (128, 0, 128),  # purple
    (0, 255, 255),  # cyan
    (255, 20, 147), # pink
    (255, 255, 0),  # yellow
    (0, 128, 0),    # dark green
    (139, 69, 19),  # brown
]


def get_color(obj_id: int) -> Tuple[int, int, int]:
    return OBJECT_COLORS[(obj_id - 1) % len(OBJECT_COLORS)]


# ---------------------------------------------------------------------------
# Mask overlay
# ---------------------------------------------------------------------------

def overlay_mask_on_frame(
    frame: np.ndarray,
    masks: Dict[int, np.ndarray],
    alpha: float = 0.45,
    draw_contour: bool = True,
) -> np.ndarray:
    """
    Blend object masks onto an RGB frame.

    Args:
        frame: (H, W, 3) uint8
        masks: {obj_id: (H, W) binary uint8 mask}
        alpha: mask transparency
        draw_contour: draw contour edges

    Returns:
        (H, W, 3) uint8 composited image
    """
    out = frame.copy().astype(np.float32)

    for obj_id, mask in masks.items():
        color = np.array(get_color(obj_id), dtype=np.float32)
        colored = np.zeros_like(out)
        colored[mask.astype(bool)] = color

        out = np.where(
            mask[:, :, None].astype(bool),
            out * (1 - alpha) + colored * alpha,
            out,
        )

        if draw_contour:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out.astype(np.uint8), contours, -1,
                             get_color(obj_id), 2)

    return out.astype(np.uint8)


def save_sequence_overlay(
    frames: List[np.ndarray],
    pred_masks: Dict[int, List[np.ndarray]],
    output_dir: str,
    sequence_name: str,
    gt_masks: Optional[Dict[int, List[np.ndarray]]] = None,
    step: int = 1,
) -> None:
    """
    Save overlaid frames to disk. If gt_masks is provided, saves side-by-side.
    """
    out_path = Path(output_dir) / sequence_name
    out_path.mkdir(parents=True, exist_ok=True)

    for f_idx in range(0, len(frames), step):
        frame = frames[f_idx]

        pred_frame_masks = {
            oid: masks[f_idx]
            for oid, masks in pred_masks.items()
            if f_idx < len(masks)
        }
        pred_overlay = overlay_mask_on_frame(frame, pred_frame_masks)

        if gt_masks is not None:
            gt_frame_masks = {
                oid: masks[f_idx]
                for oid, masks in gt_masks.items()
                if f_idx < len(masks)
            }
            gt_overlay = overlay_mask_on_frame(frame, gt_frame_masks)
            combined = np.concatenate([gt_overlay, pred_overlay], axis=1)
            cv2.imwrite(
                str(out_path / f"{f_idx:05d}.jpg"),
                cv2.cvtColor(combined, cv2.COLOR_RGB2BGR),
            )
        else:
            cv2.imwrite(
                str(out_path / f"{f_idx:05d}.jpg"),
                cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR),
            )


# ---------------------------------------------------------------------------
# Error analysis heatmap
# ---------------------------------------------------------------------------

def error_heatmap(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> np.ndarray:
    """
    Create a color-coded error map:
        Green  = True Positive
        Red    = False Positive
        Blue   = False Negative
        Black  = True Negative
    Returns (H, W, 3) uint8.
    """
    tp = np.logical_and(pred_mask, gt_mask)
    fp = np.logical_and(pred_mask, ~gt_mask.astype(bool))
    fn = np.logical_and(~pred_mask.astype(bool), gt_mask)

    out = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    out[tp] = [0, 200, 0]     # green
    out[fp] = [200, 0, 0]     # red
    out[fn] = [0, 0, 200]     # blue
    return out


def save_error_heatmap(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    frame: np.ndarray,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(frame)
    axes[0].set_title("Frame")
    axes[0].axis("off")

    overlay = overlay_mask_on_frame(frame, {1: pred_mask})
    axes[1].imshow(overlay)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    heatmap = error_heatmap(pred_mask, gt_mask)
    axes[2].imshow(heatmap)
    axes[2].set_title("Error Map (G=TP, R=FP, B=FN)")
    axes[2].axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# J&F score plots
# ---------------------------------------------------------------------------

def plot_jf_bar_chart(
    results: Dict,
    output_path: str,
    top_n: Optional[int] = None,
) -> None:
    """Bar chart of per-sequence J&F scores, sorted descending."""
    seq_names, jf_vals, j_vals, f_vals = [], [], [], []

    for seq_name, obj_dict in results["per_sequence"].items():
        j = np.mean([v["J"] for v in obj_dict.values()])
        f = np.mean([v.get("F", v["J"]) for v in obj_dict.values()])
        jf = np.mean([v["JF"] for v in obj_dict.values()])
        seq_names.append(seq_name)
        j_vals.append(j)
        f_vals.append(f)
        jf_vals.append(jf)

    # Sort by J&F
    order = np.argsort(jf_vals)[::-1]
    if top_n:
        order = order[:top_n]

    seq_names = [seq_names[i] for i in order]
    j_vals = [j_vals[i] for i in order]
    f_vals = [f_vals[i] for i in order]
    jf_vals = [jf_vals[i] for i in order]

    x = np.arange(len(seq_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(seq_names) * 0.4), 6))
    ax.bar(x - width, j_vals, width, label="J (IoU)", color="#4C72B0")
    ax.bar(x, f_vals, width, label="F (Boundary)", color="#DD8452")
    ax.bar(x + width, jf_vals, width, label="J&F", color="#55A868")
    ax.axhline(results["mean_JF"], color="red", linestyle="--",
               linewidth=1.2, label=f"Mean J&F = {results['mean_JF']:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(seq_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Per-Sequence J, F, and J&F Scores (DAVIS 2017 Validation)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved: {output_path}")


def plot_training_curves(
    log_dir: str,
    output_path: str,
) -> None:
    """
    Read TensorBoard event files and plot loss + metric curves.
    Falls back to CSV if tensorboard events are unavailable.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(log_dir)
        ea.Reload()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for tag, ax, title in [
            ("train/loss", axes[0], "Training Loss"),
            ("train/epoch_loss", axes[1], "Epoch Mean Loss"),
        ]:
            if tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                ax.plot(steps, values, linewidth=1.5)
                ax.set_title(title)
                ax.set_xlabel("Step" if "epoch" not in tag else "Epoch")
                ax.set_ylabel("Loss")
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Training curves saved: {output_path}")
    except Exception as e:
        print(f"[WARN] Could not plot training curves: {e}")


def plot_zero_shot_vs_finetuned(
    zs_results: Dict,
    ft_results: Dict,
    output_path: str,
) -> None:
    """Side-by-side comparison of zero-shot vs fine-tuned J&F per sequence."""
    seq_names = sorted(zs_results["per_sequence"].keys())

    zs_jf = [
        np.mean([v["JF"] for v in zs_results["per_sequence"][s].values()])
        for s in seq_names
    ]
    ft_jf = [
        np.mean([v["JF"] for v in ft_results["per_sequence"].get(s, {None: {"JF": 0.0}}).values()])
        for s in seq_names
    ]

    x = np.arange(len(seq_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(seq_names) * 0.4), 6))
    ax.bar(x - width / 2, zs_jf, width, label=f"Zero-shot (mean={zs_results['mean_JF']:.3f})",
           color="#4C72B0", alpha=0.85)
    ax.bar(x + width / 2, ft_jf, width,
           label=f"Fine-tuned (mean={ft_results['mean_JF']:.3f})",
           color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(seq_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("J&F Score")
    ax.set_title("Zero-Shot vs. Fine-Tuned SAM 2 on DAVIS 2017 Validation")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved: {output_path}")
