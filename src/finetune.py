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
from typing import Dict, List, Optional, Tuple

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


def fixed_weight_loss(
    logits: torch.Tensor,
    pred_iou: Optional[torch.Tensor],
    targets: torch.Tensor,
    cfg: dict,
    use_boundary: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Fixed-weight combination used by baseline and no-uncertainty runs."""
    train_cfg = cfg.get("training", {})
    indiv: Dict[str, torch.Tensor] = {
        "focal": focal_loss(logits, targets),
        "dice":  dice_loss(logits, targets),
    }
    if use_boundary:
        indiv["boundary"] = boundary_loss(logits, targets)
    if pred_iou is not None:
        indiv["iou"] = iou_loss(pred_iou, logits, targets)

    weights = {
        "focal":    train_cfg.get("focal_loss_weight", 20.0),
        "dice":     train_cfg.get("dice_loss_weight", 1.0),
        "boundary": 1.0,
        "iou":      train_cfg.get("iou_loss_weight", 1.0),
    }
    total = sum(weights[k] * v for k, v in indiv.items())
    return total, {k: v.item() for k, v in indiv.items()}


def boundary_loss(logits: torch.Tensor, targets: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Boundary-aware loss using morphological erosion via max-pooling.
    Penalises errors at mask edges to directly improve F-score."""
    pred = torch.sigmoid(logits)
    pad = kernel // 2
    pred_eroded = -F.max_pool2d(-pred,   kernel, stride=1, padding=pad)
    gt_eroded   = -F.max_pool2d(-targets, kernel, stride=1, padding=pad)
    pred_boundary = (pred   - pred_eroded).clamp(0, 1)
    gt_boundary   = (targets - gt_eroded).clamp(0, 1)
    return F.binary_cross_entropy(pred_boundary, gt_boundary)


def uncertainty_loss(
    logits: torch.Tensor,
    pred_iou: Optional[torch.Tensor],
    targets: torch.Tensor,
    log_vars: nn.ParameterDict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Kendall et al. (CVPR 2018) homoscedastic uncertainty weighting.

    Learns per-task log-variance s = log(σ²) alongside model weights.
    Derived from maximising the Gaussian/softmax log-likelihood:
      Classification terms (focal, dice, boundary): exp(-s)·L + 0.5·s
      Regression term     (iou):                  0.5·exp(-s)·L + 0.5·s
    The 0.5·s regulariser prevents σ→∞ (i.e. ignoring the task).
    """
    indiv: Dict[str, torch.Tensor] = {
        "focal":    focal_loss(logits, targets),
        "dice":     dice_loss(logits, targets),
        "boundary": boundary_loss(logits, targets),
    }
    if pred_iou is not None:
        indiv["iou"] = iou_loss(pred_iou, logits, targets)

    total = sum(
        (0.5 * torch.exp(-log_vars[k]) * v + 0.5 * log_vars[k])  # regression
        if k == "iou" else
        (torch.exp(-log_vars[k]) * v + 0.5 * log_vars[k])         # classification
        for k, v in indiv.items()
    )
    return total, {k: v.item() for k, v in indiv.items()}


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
    log_vars: Optional[nn.ParameterDict],
    prompt_type: str = "box",
    grad_accum_step: int = 1,
    grad_accum_total: int = 1,
) -> Tuple[float, Dict[str, float]]:
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
    train_cfg = cfg.get("training", {})
    use_uncertainty = train_cfg.get("use_uncertainty", True)
    use_boundary    = train_cfg.get("use_boundary", True)

    if use_uncertainty and log_vars is not None:
        loss, loss_components = uncertainty_loss(
            low_res_masks, iou_predictions, gt_tensor, log_vars
        )
    else:
        loss, loss_components = fixed_weight_loss(
            low_res_masks, iou_predictions, gt_tensor, cfg, use_boundary=use_boundary
        )

    loss = loss / grad_accum_total
    loss.backward()

    if grad_accum_step == grad_accum_total:
        max_norm = train_cfg.get("max_grad_norm", 1.0)
        all_params = [p for p in predictor.model.parameters() if p.requires_grad]
        if log_vars:
            all_params += list(log_vars.parameters())
        nn.utils.clip_grad_norm_(all_params, max_norm)
        optimizer.step()
        optimizer.zero_grad()

    return loss.item() * grad_accum_total, loss_components


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

    train_cfg = cfg.get("training", {})
    use_uncertainty = train_cfg.get("use_uncertainty", True)
    use_scheduler   = train_cfg.get("use_scheduler", True)

    trainable_params = get_trainable_parameters(predictor.model, cfg)
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check config.trainable_modules.")

    # Learnable log-variances (only for uncertainty runs)
    if use_uncertainty:
        log_var_keys = ["focal", "dice", "boundary", "iou"]
        log_vars: Optional[nn.ParameterDict] = nn.ParameterDict({
            k: nn.Parameter(torch.zeros(1, device=device)) for k in log_var_keys
        })
        print("Uncertainty log-variances initialised at 0 for:", log_var_keys)
        opt_params = list(trainable_params) + list(log_vars.parameters())
    else:
        log_vars = None
        opt_params = list(trainable_params)
        print("Fixed-weight loss (uncertainty weighting disabled).")

    optimizer = torch.optim.AdamW(
        opt_params,
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    num_epochs = train_cfg.get("num_epochs", 10)
    grad_accum = train_cfg.get("grad_accumulation_steps", 4)
    save_every = train_cfg.get("save_every_n_epochs", 2)
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/finetune"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # LR scheduler: linear warmup → cosine annealing (optional)
    if use_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        # total_steps must be in optimizer-step units (scheduler.step() is called
        # once per optimizer step, i.e. every grad_accum samples).
        total_steps  = (len(train_loader) * num_epochs) // grad_accum
        warmup_steps = train_cfg.get("warmup_steps", 100)
        warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(optimizer,
                                         T_max=max(total_steps - warmup_steps, 1),
                                         eta_min=1e-6)
        scheduler = SequentialLR(optimizer,
                                 schedulers=[warmup_sched, cosine_sched],
                                 milestones=[warmup_steps])
        print(f"Scheduler: {warmup_steps} warmup steps → cosine decay over {total_steps} optimizer steps.")
    else:
        scheduler = None
        print("Constant LR (scheduler disabled).")

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

                loss_val, loss_components = train_step(
                    predictor, frame_aug, mask_aug, optimizer, cfg,
                    log_vars=log_vars,
                    prompt_type="box",
                    grad_accum_step=accum_step,
                    grad_accum_total=grad_accum,
                )
                if accum_step == grad_accum and scheduler is not None:
                    scheduler.step()
            except Exception as e:
                print(f"[WARN] Skipping sample: {e}")
                continue

            epoch_loss += loss_val
            global_step += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}")

            # Per-step TensorBoard logging
            current_lr = (
                scheduler.get_last_lr()[0] if scheduler is not None
                else train_cfg.get("learning_rate", 1e-4)
            )
            writer.add_scalar("train/loss", loss_val, global_step)
            writer.add_scalar("train/lr", current_lr, global_step)
            for k, v in loss_components.items():
                writer.add_scalar(f"train/loss_{k}", v, global_step)
            if log_vars:
                for k in log_vars:
                    writer.add_scalar(f"train/log_var_{k}", log_vars[k].item(), global_step)

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
                    "log_vars": {k: v.item() for k, v in log_vars.items()} if log_vars else {},
                    "loss": mean_loss,
                    "cfg": cfg,
                },
                ckpt_path,
            )
            if log_vars:
                learned = {k: f"{v.item():.3f}" for k, v in log_vars.items()}
                print(f"Checkpoint saved: {ckpt_path}  |  log_vars: {learned}")
            else:
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
