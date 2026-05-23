"""
Data augmentation pipeline for SAM 2 / DAVIS 2017 training.

All transforms operate on (frame: np.ndarray, mask: np.ndarray) pairs so that
spatial transformations stay consistent between image and annotation.

Frame shape: (H, W, 3) uint8
Mask shape:  (H, W)    uint8  (0 = background, else object ID)
"""

import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance


class BrightnessJitter:
    """Randomly adjust frame brightness."""

    def __init__(self, factor: float = 0.3):
        self.factor = factor

    def __call__(
        self, frame: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pil = Image.fromarray(frame)
        delta = 1.0 + random.uniform(-self.factor, self.factor)
        pil = ImageEnhance.Brightness(pil).enhance(delta)
        return np.array(pil), mask


class ContrastJitter:
    """Randomly adjust frame contrast."""

    def __init__(self, factor: float = 0.3):
        self.factor = factor

    def __call__(
        self, frame: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pil = Image.fromarray(frame)
        delta = 1.0 + random.uniform(-self.factor, self.factor)
        pil = ImageEnhance.Contrast(pil).enhance(delta)
        return np.array(pil), mask


class RandomHorizontalFlip:
    """Flip frame and mask horizontally with given probability."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(
        self, frame: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.prob:
            frame = np.fliplr(frame).copy()
            mask = np.fliplr(mask).copy()
        return frame, mask


class RandomCrop:
    """
    Random crop keeping at least `scale[0]` fraction of the original dimensions.
    After cropping, resizes back to the original size.
    """

    def __init__(self, scale: Tuple[float, float] = (0.8, 1.0)):
        self.scale = scale

    def __call__(
        self, frame: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        scale = random.uniform(*self.scale)
        crop_h = int(h * scale)
        crop_w = int(w * scale)

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        frame_crop = frame[top : top + crop_h, left : left + crop_w]
        mask_crop = mask[top : top + crop_h, left : left + crop_w]

        # Resize back to original size
        frame_out = cv2.resize(frame_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_out = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)
        return frame_out, mask_out


class GaussianNoise:
    """Add Gaussian noise to the frame."""

    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(
        self, frame: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        noise = np.random.normal(0, self.std * 255, frame.shape).astype(np.float32)
        noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy, mask


class MotionBlur:
    """Apply motion blur to simulate fast camera or object motion."""

    def __init__(self, kernel_size: int = 15, prob: float = 0.3):
        self.kernel_size = kernel_size
        self.prob = prob

    def __call__(
        self, frame: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.prob:
            return frame, mask
        angle = random.uniform(0, 360)
        k = self.kernel_size
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        center = (k // 2, k // 2)
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (k, k))
        blurred = cv2.filter2D(frame, -1, kernel)
        return blurred, mask


class Compose:
    """Apply a sequence of augmentation transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(
        self, frame: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            frame, mask = t(frame, mask)
        return frame, mask


class SequenceCompose:
    """
    Apply the same spatial transforms to every frame in a sequence,
    and independently apply photometric transforms per-frame.

    spatial_transforms: applied with a fixed random state across all frames.
    photometric_transforms: applied independently per frame.
    """

    def __init__(
        self,
        spatial_transforms: Optional[list] = None,
        photometric_transforms: Optional[list] = None,
    ):
        self.spatial = spatial_transforms or []
        self.photometric = photometric_transforms or []

    def __call__(
        self, frames: List[np.ndarray], masks: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # Fix random seed for spatial transforms so all frames get same transform
        seed = random.randint(0, 2**31)

        out_frames, out_masks = [], []
        for frame, mask in zip(frames, masks):
            random.seed(seed)
            np.random.seed(seed)
            for t in self.spatial:
                frame, mask = t(frame, mask)

            # Photometric transforms are frame-independent
            for t in self.photometric:
                frame, mask = t(frame, mask)

            out_frames.append(frame)
            out_masks.append(mask)

        return out_frames, out_masks


def build_train_augmentation(cfg: dict) -> SequenceCompose:
    """
    Build the full augmentation pipeline from a config dict.
    Expected keys (all optional, with defaults):
        brightness_factor, horizontal_flip_prob, crop_scale,
        gaussian_noise_std, motion_blur_kernel, motion_blur_prob
    """
    a = cfg.get("augmentation", {})

    spatial = [
        RandomHorizontalFlip(prob=a.get("horizontal_flip_prob", 0.5)),
        RandomCrop(scale=tuple(a.get("crop_scale", [0.8, 1.0]))),
    ]
    photometric = [
        BrightnessJitter(factor=a.get("brightness_factor", 0.3)),
        ContrastJitter(factor=a.get("brightness_factor", 0.3)),
        GaussianNoise(std=a.get("gaussian_noise_std", 0.02)),
        MotionBlur(
            kernel_size=a.get("motion_blur_kernel", 15),
            prob=a.get("motion_blur_prob", 0.3),
        ),
    ]
    return SequenceCompose(spatial, photometric)


def build_val_augmentation() -> SequenceCompose:
    """No augmentation at validation time."""
    return SequenceCompose([], [])
