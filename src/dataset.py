"""
DAVIS 2017 dataset loader for semi-supervised Video Object Segmentation.

Expected directory layout:
    DAVIS/
    ├── Annotations/480p/{sequence}/{frame:05d}.png   (palette PNG, object IDs as pixel values)
    ├── ImageSets/2017/{train,val}.txt
    └── JPEGImages/480p/{sequence}/{frame:05d}.jpg
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DAVISSequence:
    """Single DAVIS video sequence with all frames and multi-object masks."""

    def __init__(self, sequence_name: str, davis_root: str, resolution: str = "480p"):
        self.name = sequence_name
        self.root = Path(davis_root)
        self.res = resolution

        self.frames_dir = self.root / "JPEGImages" / resolution / sequence_name
        self.masks_dir = self.root / "Annotations" / resolution / sequence_name

        frame_files = sorted(self.frames_dir.glob("*.jpg"))
        if not frame_files:
            frame_files = sorted(self.frames_dir.glob("*.png"))
        self.frame_paths = frame_files
        self.mask_paths = sorted(self.masks_dir.glob("*.png"))

        if len(self.frame_paths) == 0:
            raise FileNotFoundError(f"No frames found in {self.frames_dir}")

        # Object IDs present in frame 0 (excluding background=0)
        first_mask = np.array(Image.open(self.mask_paths[0]))
        self.object_ids = [int(v) for v in np.unique(first_mask) if v != 0]

    def __len__(self) -> int:
        return len(self.frame_paths)

    def load_frame(self, idx: int) -> np.ndarray:
        """Return RGB frame as (H, W, 3) uint8 array."""
        return np.array(Image.open(self.frame_paths[idx]).convert("RGB"))

    def load_mask(self, idx: int) -> np.ndarray:
        """Return object-ID mask as (H, W) uint8 array (0 = background)."""
        return np.array(Image.open(self.mask_paths[idx]))

    def load_binary_mask(self, idx: int, obj_id: int) -> np.ndarray:
        """Return binary mask (H, W) bool for a single object."""
        return self.load_mask(idx) == obj_id

    def get_first_frame_prompts(self) -> Dict[int, np.ndarray]:
        """Return {obj_id: binary_mask} from annotation of frame 0."""
        mask0 = self.load_mask(0)
        return {oid: (mask0 == oid) for oid in self.object_ids}


class DAVISDataset(Dataset):
    """
    PyTorch Dataset over DAVIS 2017 sequences.
    Each item is a dict with all frames and masks for one sequence.
    Used for fine-tuning (frame-level sampling) and evaluation.
    """

    def __init__(
        self,
        davis_root: str,
        split: str = "val",
        resolution: str = "480p",
        transform=None,
        max_seq_len: Optional[int] = None,
    ):
        self.root = Path(davis_root)
        self.split = split
        self.resolution = resolution
        self.transform = transform
        self.max_seq_len = max_seq_len

        split_file = self.root / "ImageSets" / "2017" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                "Make sure DAVIS 2017 is downloaded to the correct location."
            )

        with open(split_file) as f:
            self.sequence_names = [l.strip() for l in f if l.strip()]

        self.sequences = [
            DAVISSequence(name, davis_root, resolution)
            for name in self.sequence_names
        ]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        seq = self.sequences[idx]
        n = len(seq)
        if self.max_seq_len is not None:
            n = min(n, self.max_seq_len)

        frames = [seq.load_frame(i) for i in range(n)]
        masks = [seq.load_mask(i) for i in range(n)]

        if self.transform is not None:
            frames, masks = self.transform(frames, masks)

        return {
            "sequence_name": seq.name,
            "object_ids": seq.object_ids,
            "frames": frames,       # list of (H, W, 3) uint8 arrays (or tensors)
            "masks": masks,         # list of (H, W) uint8 arrays (or tensors)
            "first_frame_prompts": seq.get_first_frame_prompts(),
        }


class DAVISFrameDataset(Dataset):
    """
    Flat frame-level dataset for fine-tuning.
    Each item is a single (frame, mask, obj_id) triplet.
    """

    def __init__(
        self,
        davis_root: str,
        split: str = "train",
        resolution: str = "480p",
        transform=None,
    ):
        self.transform = transform
        seq_dataset = DAVISDataset(davis_root, split, resolution)
        self.samples: List[Tuple[DAVISSequence, int, int]] = []

        for seq in seq_dataset.sequences:
            for frame_idx in range(len(seq)):
                for obj_id in seq.object_ids:
                    self.samples.append((seq, frame_idx, obj_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        seq, frame_idx, obj_id = self.samples[idx]
        frame = seq.load_frame(frame_idx)
        mask = seq.load_binary_mask(frame_idx, obj_id).astype(np.uint8)

        if self.transform is not None:
            frame, mask = self.transform(frame, mask)

        return {
            "sequence_name": seq.name,
            "frame_idx": frame_idx,
            "obj_id": obj_id,
            "frame": frame,
            "mask": mask,
        }


def get_gt_masks_for_evaluation(
    davis_root: str,
    split: str = "val",
    resolution: str = "480p",
) -> Dict[str, Dict[int, List[np.ndarray]]]:
    """
    Load all ground-truth masks organised as:
        {sequence_name: {obj_id: [mask_frame_0, mask_frame_1, ...]}}
    Used for J&F score computation.
    """
    ds = DAVISDataset(davis_root, split, resolution)
    gt = {}
    for seq in ds.sequences:
        gt[seq.name] = {
            oid: [seq.load_binary_mask(i, oid).astype(np.uint8) for i in range(len(seq))]
            for oid in seq.object_ids
        }
    return gt
