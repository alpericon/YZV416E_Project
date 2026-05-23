"""
Fine-tuning SAM 2 on DAVIS 2017 using a frozen backbone (MedSAM-style adaptation).

Strategy:
    - Freeze: image encoder (Hiera ViT backbone)
    - Train:  memory_attention, sam_mask_decoder
    - Loss:   Focal + Dice (+ IoU prediction)
    - Inputs: single annotated frames from DAVIS training set

Training loop uses SAM2ImagePredictor which gives direct access to the model
for gradient computation. Each step = one (frame, mask, prompt) triplet.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import DAVISFrameDataset
from .augmentation import build_train_augmentation, build_val_augmentation
from .preprocessing import (
    preprocess_frame_tensor,
    mask_to_tensor,
    mask_to_box,
    mask_to_point,
    SAM2_INPUT_SIZE,
)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1.0 - pt) ** gamma * bce).mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    numerator = 2.0 * (probs * targets).sum(dim=(-2, -1))
    denominator = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    return (1.0 - numerator / (denominator + 1e-6)).mean()


def iou_loss(pred_iou: torch.Tensor, logits: torch.Tensor,
             targets: torch.Tensor) -> torch.Tensor:
    """Match predicted IoU score to actual IoU."""
    with torch.no_grad():
        probs = (torch.sigmoid(logits) > 0.5).float()
        intersection = (probs * targets).sum(dim=(-2, -1))
        union = (probs + targets).clamp(0, 1).sum(dim=(-2, -1))
        true_iou = intersection / (union + 1e-6)  # (B, num_masks)
    return F.mse_loss(pred_iou, true_iou.view_as(pred_iou))


def combined_loss(logits, pred_iou, targets, cfg: dict) -> torch.Tensor:
    w = cfg.get("training", {})
    focal_w = w.get("focal_loss_weight", 20.0)
    dice_w = w.get("dice_loss_weight", 1.0)
    iou_w = w.get("iou_loss_weight", 1.0)

    loss = focal_w * focal_loss(logits, targets)
    loss = loss + dice_w * dice_loss(logits, targets)
    if pred_iou is not None:
        loss = loss + iou_w * iou_loss(pred_iou, logits, targets)
    return loss


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_backbone(model) -> None:
    """Freeze the image encoder parameters."""
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    print("Image encoder frozen.")


def get_trainable_parameters(model, cfg: dict) -> List[nn.Parameter]:
    """Return parameters of modules listed in cfg.training.trainable_modules."""
    trainable_modules = cfg.get("training", {}).get(
        "trainable_modules", ["memory_attention", "sam_mask_decoder"]
    )
    params = []
    for module_name in trainable_modules:
        module = getattr(model, module_name, None)
        if module is None:
            print(f"[WARN] Module '{module_name}' not found in model — skipping.")
            continue
        count = sum(p.numel() for p in module.parameters())
        print(f"  Trainable: {module_name} ({count:,} parameters)")
        params.extend(module.parameters())
    return params


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def _encode_image_no_grad(predictor, frame: np.ndarray, device: torch.device):
    """
    Run the frozen image encoder and return cached feature dicts.
    Equivalent to predictor.set_image() but explicitly under no_grad.
    """
    input_tensor = predictor._transforms(frame)          # (3, H, W)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
    orig_hw = frame.shape[:2]

    with torch.no_grad():
        backbone_out = predictor.model.forward_image(input_tensor)
        _, vision_feats, _, _ = predictor.model._prepare_backbone_features(backbone_out)
        if predictor.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + predictor.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], predictor._bb_feat_sizes[::-1])
        ][::-1]

    return (
        feats[-1],    # image_embed  (1, C, 64, 64)
        feats[:-1],   # high_res_feats  [(1,C,256,256), (1,C,128,128)]
        orig_hw,
    )


def train_step(
    predictor,
    frame: np.ndarray,
    gt_mask: np.ndarray,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    prompt_type: str = "box",
    grad_accum_step: int = 1,
    grad_accum_total: int = 1,
) -> float:
    """
    One gradient step on a single (frame, mask) pair.

    Calls the model components directly so that gradients flow through
    sam_prompt_encoder and sam_mask_decoder (both trainable), while the
    image encoder runs under torch.no_grad() (frozen).
    """
    device = next(predictor.model.parameters()).device

    # ---- 1. Image encoder (frozen, no grad) --------------------------------
    image_embed, high_res_feats, orig_hw = _encode_image_no_grad(
        predictor, frame, device
    )

    # ---- 2. Build GT mask tensor at low-res decoder output (256×256) -------
    # SAM 2 decoder outputs at 1/4 of the model input size → 256×256
    low_res_size = predictor.model.image_size // 4  # typically 256
    import cv2
    gt_lr = cv2.resize(
        gt_mask.astype(np.uint8), (low_res_size, low_res_size),
        interpolation=cv2.INTER_NEAREST,
    )
    gt_tensor = torch.from_numpy(gt_lr).float().unsqueeze(0).unsqueeze(0).to(device)
    # (1, 1, 256, 256)

    # ---- 3. Build box prompt in model coordinate space ---------------------
    # Use the full-res GT mask to derive the box, then transform coords
    box_np = mask_to_box(gt_mask.astype(np.uint8))   # [x1,y1,x2,y2] in orig space
    box_t  = torch.as_tensor(box_np, dtype=torch.float, device=device).unsqueeze(0)  # (1,4)
    # Transform to SAM 2 model space (normalised / resized coords)
    box_t = predictor._transforms.transform_boxes(
        box_t, normalize=True, orig_hw=orig_hw
    )  # (1, 2, 2)

    # Encode as concat_points (SAM 2 treats box corners as special point types)
    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device)  # (1, 2)
    concat_points = (box_t, box_labels)   # coords (1,2,2), labels (1,2)

    # ---- 4. Prompt encoder + mask decoder (trainable, with grad) -----------
    sparse_emb, dense_emb = predictor.model.sam_prompt_encoder(
        points=concat_points,
        boxes=None,
        masks=None,
    )

    low_res_masks, iou_predictions, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_feats,
    )
    # low_res_masks: (1, 1, 256, 256)   iou_predictions: (1, 1)

    # ---- 5. Loss & backward ------------------------------------------------
    loss = combined_loss(low_res_masks, iou_predictions, gt_tensor, cfg)
    loss = loss / grad_accum_total
    loss.backward()

    if grad_accum_step == grad_accum_total:
        max_norm = cfg.get("training", {}).get("max_grad_norm", 1.0)
        nn.utils.clip_grad_norm_(
            [p for p in predictor.model.parameters() if p.requires_grad], max_norm
        )
        optimizer.step()
        optimizer.zero_grad()

    return loss.item() * grad_accum_total


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    """Run the full fine-tuning loop."""
    from .inference import load_sam2_image_predictor

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build datasets
    aug = build_train_augmentation(cfg)

    def frame_transform(frame, mask):
        return aug([frame], [mask])[0][0], aug([frame], [mask])[1][0]

    train_dataset = DAVISFrameDataset(
        cfg["data"]["davis_root"],
        split=cfg["data"].get("train_split", "train"),
        transform=None,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=lambda x: x[0],
    )

    # Load model
    predictor = load_sam2_image_predictor(cfg)
    predictor.model.to(device)

    # Freeze backbone
    freeze_backbone(predictor.model)

    # Optimizer
    trainable_params = get_trainable_parameters(predictor.model, cfg)
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check config.trainable_modules.")

    train_cfg = cfg.get("training", {})
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    num_epochs = train_cfg.get("num_epochs", 10)
    grad_accum = train_cfg.get("grad_accumulation_steps", 4)
    save_every = train_cfg.get("save_every_n_epochs", 2)
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/finetune"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(ckpt_dir / "tensorboard"))
    global_step = 0
    optimizer.zero_grad()

    print(f"\nStarting fine-tuning for {num_epochs} epochs on {len(train_dataset)} frames.")
    print(f"Effective batch size: {grad_accum} (grad accumulation)\n")

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        accum_step = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for sample in pbar:
            frame = sample["frame"]
            mask = sample["mask"]
            accum_step = (accum_step % grad_accum) + 1

            try:
                # Apply augmentations inline (they're not in the DataLoader)
                frame_aug, mask_aug = frame, mask
                if aug is not None:
                    frames_out, masks_out = aug([frame_aug], [mask_aug])
                    frame_aug, mask_aug = frames_out[0], masks_out[0]

                loss_val = train_step(
                    predictor, frame_aug, mask_aug, optimizer, cfg,
                    prompt_type="box",
                    grad_accum_step=accum_step,
                    grad_accum_total=grad_accum,
                )
            except Exception as e:
                print(f"[WARN] Skipping sample: {e}")
                continue

            epoch_loss += loss_val
            global_step += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}")
            writer.add_scalar("train/loss", loss_val, global_step)

        mean_loss = epoch_loss / max(len(train_dataset), 1)
        print(f"Epoch {epoch} — mean loss: {mean_loss:.6f}")
        writer.add_scalar("train/epoch_loss", mean_loss, epoch)

        if epoch % save_every == 0 or epoch == num_epochs:
            ckpt_path = ckpt_dir / f"sam2_finetune_epoch{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": predictor.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": mean_loss,
                    "cfg": cfg,
                },
                ckpt_path,
            )
            print(f"Checkpoint saved: {ckpt_path}")

    writer.close()
    print("\nFine-tuning complete.")


def load_finetuned_predictor(cfg: dict, checkpoint_path: str):
    """
    Load fine-tuned weights into a SAM2VideoPredictor for evaluation.

    Fine-tuning saves model.state_dict() from SAM2Base (shared by both
    image and video predictors). Evaluation requires the video predictor
    so we build that and then overwrite its weights from the checkpoint.
    """
    from .inference import load_sam2_video_predictor

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    predictor = load_sam2_video_predictor(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # SAM2VideoPredictor IS the model (extends SAM2Base directly), so load
    # into predictor itself — not predictor.model as in SAM2ImagePredictor.
    predictor.load_state_dict(ckpt["model_state_dict"])
    predictor.to(device)
    print(f"Loaded fine-tuned checkpoint: {checkpoint_path}  (epoch {ckpt['epoch']})")
    return predictor
