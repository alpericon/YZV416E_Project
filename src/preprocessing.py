"""
Preprocessing utilities for SAM 2 / DAVIS 2017.

Handles:
- ImageNet normalization
- Frame resizing to 1024x1024
- Compressed Run-Length Encoding (RLE) for masks (COCO format via pycocotools)
- Tensor conversion helpers
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ImageNet statistics (used by SAM 2's image encoder)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SAM2_INPUT_SIZE = 1024


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def resize_frame(frame: np.ndarray, size: int = SAM2_INPUT_SIZE) -> np.ndarray:
    """Resize (H, W, 3) uint8 frame to (size, size, 3) using LANCZOS."""
    pil = Image.fromarray(frame)
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil)


def resize_mask(mask: np.ndarray, size: int = SAM2_INPUT_SIZE) -> np.ndarray:
    """Resize (H, W) uint8 mask to (size, size) using NEAREST (preserves labels)."""
    pil = Image.fromarray(mask)
    pil = pil.resize((size, size), Image.NEAREST)
    return np.array(pil)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convert (H, W, 3) uint8 [0, 255] → float32 [0, 1] and apply ImageNet stats.
    Returns (H, W, 3) float32.
    """
    img = frame.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def preprocess_frame(frame: np.ndarray, size: int = SAM2_INPUT_SIZE) -> np.ndarray:
    """Resize + normalize a single frame. Returns (H, W, 3) float32."""
    frame = resize_frame(frame, size)
    return normalize_frame(frame)


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert (H, W, 3) float32 ndarray → (3, H, W) float32 tensor."""
    return torch.from_numpy(frame).permute(2, 0, 1).float()


def preprocess_frame_tensor(
    frame: np.ndarray, size: int = SAM2_INPUT_SIZE
) -> torch.Tensor:
    """Resize, normalize, and convert to (3, H, W) float32 tensor."""
    return frame_to_tensor(preprocess_frame(frame, size))


def preprocess_sequence(
    frames: List[np.ndarray], size: int = SAM2_INPUT_SIZE
) -> torch.Tensor:
    """
    Preprocess a list of frames.
    Returns (T, 3, H, W) float32 tensor.
    """
    tensors = [preprocess_frame_tensor(f, size) for f in frames]
    return torch.stack(tensors)


# ---------------------------------------------------------------------------
# Mask preprocessing
# ---------------------------------------------------------------------------

def preprocess_mask(mask: np.ndarray, size: int = SAM2_INPUT_SIZE) -> np.ndarray:
    """Resize mask to (size, size) uint8 using nearest-neighbor."""
    return resize_mask(mask, size)


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """Convert (H, W) uint8 binary mask → (1, H, W) float32 tensor."""
    return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)


# ---------------------------------------------------------------------------
# RLE encoding / decoding (COCO format via pycocotools)
# ---------------------------------------------------------------------------

def encode_mask_rle(binary_mask: np.ndarray) -> Dict:
    """
    Encode a binary mask to COCO RLE format.
    Input: (H, W) bool or uint8 array.
    Output: {'size': [H, W], 'counts': <bytes>}
    """
    try:
        from pycocotools import mask as coco_mask
        mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
        return coco_mask.encode(mask_fortran)
    except ImportError:
        return _encode_rle_numpy(binary_mask)


def decode_mask_rle(rle: Dict) -> np.ndarray:
    """
    Decode COCO RLE back to a (H, W) uint8 binary mask.
    """
    try:
        from pycocotools import mask as coco_mask
        return coco_mask.decode(rle).astype(np.uint8)
    except ImportError:
        return _decode_rle_numpy(rle)


def _encode_rle_numpy(mask: np.ndarray) -> Dict:
    """Pure-numpy fallback RLE encoder (row-major order)."""
    flat = mask.flatten(order="F").astype(np.uint8)
    changes = np.diff(np.concatenate([[0], flat, [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    counts = []
    prev = 0
    for s, e in zip(starts, ends):
        counts.append(int(s - prev))
        counts.append(int(e - s))
        prev = e
    counts.append(int(len(flat) - prev))
    return {"size": list(mask.shape), "counts": counts}


def _decode_rle_numpy(rle: Dict) -> np.ndarray:
    """Pure-numpy fallback RLE decoder."""
    h, w = rle["size"]
    counts = rle["counts"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2 == 1:
            flat[pos : pos + c] = 1
        pos += c
    return flat.reshape((h, w), order="F")


def encode_sequence_masks(
    masks: List[np.ndarray],
) -> List[Dict]:
    """Encode a list of binary masks to RLE."""
    return [encode_mask_rle(m) for m in masks]


# ---------------------------------------------------------------------------
# Prompt utilities (for SAM 2 input preparation)
# ---------------------------------------------------------------------------

def mask_to_box(binary_mask: np.ndarray) -> np.ndarray:
    """Extract tight bounding box [x1, y1, x2, y2] from a binary mask."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any():
        return np.array([0, 0, 1, 1], dtype=np.float32)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def mask_to_point(binary_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a single positive point (centroid of mask).
    Returns (points, labels) where points is (1, 2) [x, y] and labels is (1,).
    """
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        cx, cy = binary_mask.shape[1] // 2, binary_mask.shape[0] // 2
    else:
        cx, cy = int(xs.mean()), int(ys.mean())
        if not binary_mask[cy, cx]:
            idx = len(xs) // 2
            cx, cy = int(xs[idx]), int(ys[idx])
    return np.array([[cx, cy]], dtype=np.float32), np.array([1], dtype=np.int32)


def scale_prompt_to_resolution(
    prompt: np.ndarray,
    original_size: Tuple[int, int],
    target_size: int = SAM2_INPUT_SIZE,
) -> np.ndarray:
    """
    Scale box [x1, y1, x2, y2] or point coordinates from original image
    resolution to target resolution.
    """
    orig_h, orig_w = original_size
    scale_x = target_size / orig_w
    scale_y = target_size / orig_h
    scaled = prompt.copy().astype(np.float32)
    scaled[..., 0] *= scale_x   # x coords
    scaled[..., 1] *= scale_y   # y coords
    if scaled.shape[-1] == 4:   # box: also scale x2, y2
        scaled[..., 2] *= scale_x
        scaled[..., 3] *= scale_y
    return scaled
