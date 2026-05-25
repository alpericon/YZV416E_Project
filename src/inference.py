"""
SAM 2 zero-shot and prompted Video Object Segmentation inference.

Supports three prompt types on the first frame:
    - 'mask'  : use the ground-truth binary mask from frame 0
    - 'box'   : derive a tight bounding box from the GT mask
    - 'point' : derive the centroid point from the GT mask

Usage:
    predictor = SAM2VideoPredictor.load(cfg)
    results = run_zero_shot_on_dataset(predictor, davis_root, split, cfg)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .dataset import DAVISDataset, DAVISSequence
from .preprocessing import (
    mask_to_box,
    mask_to_point,
    resize_mask,
    SAM2_INPUT_SIZE,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_sam2_video_predictor(cfg: dict):
    """
    Load SAM2VideoPredictor from either a local checkpoint or Hugging Face.

    cfg keys used:
        model.checkpoint  — path to local .pt file
        model.config      — SAM 2 yaml config name (e.g. 'sam2_hiera_b+.yaml')
        model.hf_model_id — HF model ID (fallback if checkpoint missing)
        device            — 'cuda' or 'cpu'
    """
    from sam2.build_sam import build_sam2_video_predictor

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    checkpoint = model_cfg.get("checkpoint", "")
    config = model_cfg.get("config", "sam2_hiera_b+.yaml")

    if os.path.exists(checkpoint):
        print(f"Loading SAM 2 from local checkpoint: {checkpoint}")
        predictor = build_sam2_video_predictor(config, checkpoint, device=device)
    else:
        hf_id = model_cfg.get("hf_model_id", "facebook/sam2-hiera-base-plus")
        print(f"Checkpoint not found locally. Loading from HuggingFace: {hf_id}")
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        predictor = SAM2VideoPredictor.from_pretrained(hf_id, device=device)

    return predictor


def load_sam2_image_predictor(cfg: dict):
    """
    Load SAM2ImagePredictor — used during fine-tuning (frame-level).
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    checkpoint = model_cfg.get("checkpoint", "")
    config = model_cfg.get("config", "sam2_hiera_b+.yaml")

    if os.path.exists(checkpoint):
        model = build_sam2(config, checkpoint, device=device)
    else:
        hf_id = model_cfg.get("hf_model_id", "facebook/sam2-hiera-base-plus")
        print(f"Checkpoint not found locally. Loading from HuggingFace: {hf_id}")
        return SAM2ImagePredictor.from_pretrained(hf_id, device=device)

    return SAM2ImagePredictor(model)


# ---------------------------------------------------------------------------
# Single-sequence inference
# ---------------------------------------------------------------------------

def run_inference_on_sequence(
    predictor,
    sequence: DAVISSequence,
    prompt_type: str = "mask",
    target_size: int = SAM2_INPUT_SIZE,
) -> Dict[int, List[np.ndarray]]:
    """
    Run SAM 2 video inference on a single DAVIS sequence.

    Returns:
        {obj_id: [binary_mask_frame_0, binary_mask_frame_1, ...]}
        Masks are resized back to the original frame resolution.
    """
    import tempfile, shutil

    orig_h, orig_w = sequence.load_frame(0).shape[:2]
    first_frame_prompts = sequence.get_first_frame_prompts()

    # SAM 2 VideoPredictor works from a directory of JPEG frames
    with torch.inference_mode(), torch.autocast(
        "cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16
    ):
        state = predictor.init_state(
            video_path=str(sequence.frames_dir),
            offload_video_to_cpu=True,
            async_loading_frames=False,
        )
        predictor.reset_state(state)

        # Register prompts for each object on frame 0.
        # Prompts must be in the original video resolution — SAM 2 video
        # predictor handles internal resizing itself.
        for obj_id, gt_mask in first_frame_prompts.items():
            gt_mask_orig = gt_mask.astype(np.uint8)  # keep at original (H, W)

            if prompt_type == "mask":
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=gt_mask_orig.astype(bool),
                )
            elif prompt_type == "box":
                box = mask_to_box(gt_mask_orig)  # coords in original space
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=box,
                )
            elif prompt_type == "point":
                points, labels = mask_to_point(gt_mask_orig)  # coords in original space
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )
            else:
                raise ValueError(f"Unknown prompt_type: {prompt_type}")

        # Propagate through all frames.
        # propagate_in_video returns masks already upscaled to original video
        # resolution by SAM 2 — no further resizing needed.
        pred_masks: Dict[int, List[np.ndarray]] = {
            oid: [] for oid in first_frame_prompts
        }

        for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
            for oid, mask in zip(obj_ids, masks):
                # mask: (1, H, W) at original video resolution
                binary = (mask[0] > 0.0).cpu().numpy().astype(np.uint8)
                pred_masks[int(oid)].append(binary)

    return pred_masks


# ---------------------------------------------------------------------------
# Full dataset inference
# ---------------------------------------------------------------------------

def run_zero_shot_on_dataset(
    predictor,
    davis_root: str,
    split: str = "val",
    prompt_type: str = "mask",
    cfg: Optional[dict] = None,
) -> Dict[str, Dict[int, List[np.ndarray]]]:
    """
    Run zero-shot inference over the full DAVIS split.

    Returns:
        {seq_name: {obj_id: [binary_mask_per_frame]}}
    """
    dataset = DAVISDataset(davis_root, split=split)
    results = {}

    for seq in tqdm(dataset.sequences, desc=f"Inference [{split}]"):
        try:
            pred = run_inference_on_sequence(predictor, seq, prompt_type=prompt_type)
            results[seq.name] = pred
        except Exception as e:
            print(f"[WARN] Skipping {seq.name}: {e}")
            results[seq.name] = {}

    return results


# ---------------------------------------------------------------------------
# Save / load predictions
# ---------------------------------------------------------------------------

def save_predictions(
    predictions: Dict[str, Dict[int, List[np.ndarray]]],
    output_dir: str,
) -> None:
    """
    Save predicted masks as PNG files in DAVIS-compatible structure:
        output_dir/{sequence_name}/{frame:05d}.png
    Multi-object sequences are merged into a single palette PNG.
    """
    from PIL import Image

    for seq_name, obj_dict in predictions.items():
        seq_dir = Path(output_dir) / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        if not obj_dict:
            continue

        num_frames = max(len(m) for m in obj_dict.values())
        for f_idx in range(num_frames):
            # Build combined mask: each object ID occupies its pixel value
            sample_mask = next(iter(obj_dict.values()))[f_idx]
            combined = np.zeros_like(sample_mask, dtype=np.uint8)

            for obj_id, masks in obj_dict.items():
                if f_idx < len(masks):
                    combined[masks[f_idx].astype(bool)] = obj_id

            pil = Image.fromarray(combined, mode="P")
            pil.save(seq_dir / f"{f_idx:05d}.png")


def load_predictions(
    prediction_dir: str,
    sequence_names: List[str],
    object_ids_map: Dict[str, List[int]],
) -> Dict[str, Dict[int, List[np.ndarray]]]:
    """Load saved predictions back into memory."""
    from PIL import Image

    predictions = {}
    for seq_name in sequence_names:
        seq_dir = Path(prediction_dir) / seq_name
        mask_files = sorted(seq_dir.glob("*.png"))
        obj_ids = object_ids_map.get(seq_name, [])

        predictions[seq_name] = {oid: [] for oid in obj_ids}
        for mf in mask_files:
            combined = np.array(Image.open(mf))
            for oid in obj_ids:
                predictions[seq_name][oid].append((combined == oid).astype(np.uint8))

    return predictions
