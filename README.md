# SAMet — SAM 2 Video Object Segmentation
**YZV416E Computer Vision | Istanbul Technical University | Spring 2026**
Team: Alper Düzgün · Enis Furkan Kırmızı · Akın Yağbasan

---

## Requirements
- Python 3.10+
- Git
- NVIDIA GPU (8 GB+ VRAM recommended)
- CUDA 11.8+

---

## Step 1 — Clone this repo and enter the folder

```cmd
cd "C:\Users\alper\Desktop\computer vision\proje"
```

---

## Step 2 — Create virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

> Every time you open a new terminal, run `.venv\Scripts\activate` before anything else.

---

## Step 3 — Install PyTorch (CUDA 11.8)

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 4 — Install project dependencies

```cmd
pip install -r requirements.txt
```

---

## Step 5 — Install SAM 2

```cmd
git clone https://github.com/facebookresearch/sam2.git sam2_repo
cd sam2_repo
pip install -e ".[demo]"
cd ..
```

---

## Step 6 — Install DAVIS evaluation toolkit

```cmd
pip install git+https://github.com/davisvideochallenge/davis2017-evaluation.git
```

---

## Step 7 — Download SAM 2 checkpoint

Download `sam2_hiera_base_plus.pt` and place it in the `checkpoints/` folder.

```
checkpoints/
└── sam2_hiera_base_plus.pt
```

Via the download script (downloads checkpoint only, skips DAVIS):
```cmd
python scripts/download_davis.py --skip-davis
```

Or via Python (recommended on Kaggle — avoids stale CDN URLs):
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="facebook/sam2-hiera-base-plus",
    filename="sam2_hiera_base_plus.pt",
    local_dir="checkpoints",
)
```

Direct download link (from Meta):
```
https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
```

---

## Step 8 — Download DAVIS 2017 dataset

```cmd
python scripts/download_davis.py --skip-checkpoint
```

This downloads and extracts DAVIS 2017 into `data/DAVIS/`. Expected structure:

```
data/DAVIS/
├── Annotations/480p/{sequence}/{frame:05d}.png
├── ImageSets/2017/train.txt
├── ImageSets/2017/val.txt
└── JPEGImages/480p/{sequence}/{frame:05d}.jpg
```

---

## Step 9 — Verify GPU is available

```cmd
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"
```

---

## Step 10 — Run zero-shot evaluation (baseline)

Evaluates SAM 2 out-of-the-box on DAVIS 2017 val set using first-frame mask prompt.

```cmd
python scripts/run_zero_shot.py
```

Results saved to `results/zero_shot_mask/`:
- `results.csv` — per-sequence J, F, J&F scores
- `results_summary.json` — mean scores
- `jf_bar_chart.png` — bar chart visualization
- `visualizations/` — side-by-side overlays (GT vs prediction)

**Expected result: ~0.905 mean J&F**

---

## Step 11 — Run fine-tuning (ablation runs)

Fine-tunes the SAM 2 mask decoder on DAVIS 2017 training set (backbone frozen). Three experimental configurations are supported.

### Default (improved — all enhancements enabled)
Boundary-aware loss + homoscedastic uncertainty weighting (Kendall et al. CVPR 2018) + LR scheduler.
```cmd
python scripts/run_finetune.py --epochs 10
```
Checkpoints: `checkpoints/finetune_improved/`
Results: `results/finetune_improved/`

### Baseline (constant LR, focal+dice+iou loss only)
```cmd
python scripts/run_finetune.py --baseline --epochs 10
```
Checkpoints: `checkpoints/finetune_baseline/`
Results: `results/finetune_baseline/`

### Ablation: no uncertainty weighting (scheduler + boundary loss, fixed weights)
```cmd
python scripts/run_finetune.py --no-uncertainty --epochs 10
```
Checkpoints: `checkpoints/finetune_no_uncertainty/`
Results: `results/finetune_no_uncertainty/`

### Options
```cmd
# Change number of epochs
python scripts/run_finetune.py --epochs 10

# Change learning rate
python scripts/run_finetune.py --epochs 10 --lr 1e-5

# Skip training, only run evaluation on a saved checkpoint
python scripts/run_finetune.py --eval-only --checkpoint checkpoints\finetune_improved\sam2_finetune_epoch010.pt

# Custom run name (checkpoint and output dirs use this label)
python scripts/run_finetune.py --run-name myrun --epochs 5
```

Each run produces in its results directory:
- `results_finetuned.csv`
- `results_finetuned_summary.json`
- `jf_bar_chart_finetuned.png`
- `training_curves.png`
- `visualizations/`

---

## Step 12 — Monitor training with TensorBoard

Each run writes TensorBoard events to its own checkpoint subdirectory:

```cmd
# Default (improved) run
tensorboard --logdir checkpoints/finetune_improved/tensorboard

# Baseline run
tensorboard --logdir checkpoints/finetune_baseline/tensorboard

# All runs at once (compare)
tensorboard --logdir checkpoints
```

Then open `http://localhost:6006` in your browser.

---

## Full pipeline (all ablation runs)

```cmd
# Zero-shot baseline
python scripts/run_zero_shot.py

# Fine-tuning: all three ablation variants
python scripts/run_finetune.py --baseline --epochs 10
python scripts/run_finetune.py --no-uncertainty --epochs 10
python scripts/run_finetune.py --epochs 10
```

---

## Results Summary

| Model | Mean J | Mean F | Mean J&F |
|---|---|---|---|
| SAM 2 Zero-shot (mask prompt) | 0.855 | 0.956 | **0.905** |
| Fine-tuned decoder — baseline (4 epochs) | 0.796 | 0.942 | **0.869** |
| Fine-tuned decoder — no uncertainty (10 epochs) | TBD | TBD | TBD |
| Fine-tuned decoder — improved (10 epochs) | TBD | TBD | TBD |

### Hardest sequences (both models)
| Sequence | Challenge |
|---|---|
| kite-surf | Thin structures, low contrast |
| paragliding-launch | Similar-looking objects |
| lab-coat | Identity switching (5 similar people) |

---

## Project structure

```
proje/
├── configs/
│   └── config.yaml                   # All hyperparameters
├── checkpoints/
│   ├── sam2_hiera_base_plus.pt
│   ├── finetune_improved/            # Improved run checkpoints + tensorboard
│   ├── finetune_baseline/            # Baseline run checkpoints + tensorboard
│   └── finetune_no_uncertainty/      # No-uncertainty run checkpoints + tensorboard
├── data/
│   └── DAVIS/                        # Dataset (downloaded)
├── results/
│   ├── zero_shot_mask/               # Zero-shot evaluation outputs
│   ├── finetune_improved/            # Improved run evaluation outputs
│   ├── finetune_baseline/            # Baseline run evaluation outputs
│   └── finetune_no_uncertainty/      # No-uncertainty run evaluation outputs
├── sam2_repo/                        # SAM 2 source (cloned)
├── scripts/
│   ├── download_davis.py             # Dataset + checkpoint downloader
│   ├── run_zero_shot.py              # Zero-shot evaluation
│   └── run_finetune.py               # Fine-tuning + evaluation (3 ablation modes)
├── src/
│   ├── dataset.py                    # DAVIS 2017 data loader
│   ├── preprocessing.py              # Normalization, resize, RLE
│   ├── augmentation.py               # Training augmentations
│   ├── inference.py                  # SAM 2 video inference
│   ├── finetune.py                   # Fine-tuning loop + loss functions
│   ├── evaluate.py                   # J&F score computation
│   └── visualize.py                  # Plots and overlays
├── notebooks/
│   └── 01_baseline.ipynb             # Interactive baseline notebook
├── requirements.txt
├── setup.bat                         # Windows automated setup
└── setup.sh                          # Linux/HPC automated setup
```

---

## Config reference (`configs/config.yaml`)

| Key | Default | Description |
|---|---|---|
| `model.config` | `configs/sam2/sam2_hiera_b+.yaml` | SAM 2 architecture config |
| `model.checkpoint` | `checkpoints/sam2_hiera_base_plus.pt` | Model weights |
| `training.trainable_modules` | `[sam_mask_decoder]` | Which layers to fine-tune (backbone always frozen) |
| `training.learning_rate` | `1e-4` | AdamW learning rate |
| `training.num_epochs` | `10` | Training epochs |
| `training.grad_accumulation_steps` | `4` | Effective batch size multiplier |
| `training.warmup_steps` | `100` | Linear warmup steps before cosine annealing |
| `training.use_uncertainty` | `true` | Kendall et al. homoscedastic uncertainty weighting |
| `training.use_boundary` | `true` | Boundary-aware loss (improves F-score) |
| `training.use_scheduler` | `true` | LR warmup → cosine annealing |
| `evaluation.split` | `val` | DAVIS split to evaluate on |

> These flags are set automatically by `--baseline` and `--no-uncertainty`; you do not need to edit `config.yaml` manually.

---

## Troubleshooting

**`SAM2Base.__init__() missing arguments`**
→ Wrong config name. Make sure `model.config` in `config.yaml` is `configs/sam2/sam2_hiera_b+.yaml`

**`element 0 of tensors does not require grad`**
→ Fixed. The training step now calls model components directly instead of `predictor.predict()`.

**`'SAM2VideoPredictor' object has no attribute 'model'`**
→ Fixed. `load_finetuned_predictor` uses `predictor.load_state_dict()` directly (SAM2VideoPredictor IS the model).

**Masks all black / J&F near zero**
→ Fixed. Mask prompts must be passed at original video resolution (480×854), not resized to 1024×1024.

**NaN loss during training**
→ Fixed. Boundary loss now clamps `pred_boundary` to `[1e-6, 1-1e-6]` before BCE to prevent `log(0)`.

**Out of memory (CUDA OOM)**
→ Switch to ViT-T in `config.yaml`:
```yaml
model:
  variant: hiera_t
  config: configs/sam2/sam2_hiera_t.yaml
  checkpoint: checkpoints/sam2_hiera_tiny.pt
```

**No checkpoints found after training**
→ Checkpoints are now saved to `checkpoints/finetune_{run_name}/` (e.g. `finetune_improved`), not `checkpoints/finetune/`.
