"""
Fine-tune SAM 2 on DAVIS 2017 training set, then evaluate on validation set.

Usage:
    python scripts/run_finetune.py
    python scripts/run_finetune.py --config configs/config.yaml --epochs 10
    python scripts/run_finetune.py --eval-only --checkpoint checkpoints/finetune/sam2_finetune_epoch010.pt
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import DAVISDataset, get_gt_masks_for_evaluation
from src.evaluate import evaluate_dataset, print_results, save_results_csv
from src.finetune import train, load_finetuned_predictor
from src.inference import (
    load_sam2_video_predictor,
    run_zero_shot_on_dataset,
    save_predictions,
)
from src.visualize import (
    plot_jf_bar_chart,
    plot_training_curves,
    plot_zero_shot_vs_finetuned,
    save_sequence_overlay,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SAM 2 on DAVIS 2017")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override num_epochs from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning_rate from config")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; only run evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned checkpoint for eval-only mode")
    parser.add_argument("--compare-zero-shot", action="store_true", default=True,
                        help="Also run zero-shot eval and compare (default: true)")
    parser.add_argument("--no-compare", dest="compare_zero_shot", action="store_false")
    parser.add_argument(
        "--run-name", default="improved",
        help="Label for this run; controls checkpoint and output subdirs (default: improved)"
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Baseline run: constant LR, no boundary loss, no uncertainty weighting"
    )
    parser.add_argument(
        "--no-uncertainty", dest="no_uncertainty", action="store_true",
        help="Ablation: LR scheduler + boundary loss, but no uncertainty weighting"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.epochs is not None:
        cfg.setdefault("training", {})["num_epochs"] = args.epochs
    if args.lr is not None:
        cfg.setdefault("training", {})["learning_rate"] = args.lr

    # Run variant flags — determine feature set and resolve run name
    run_name = args.run_name
    t = cfg.setdefault("training", {})
    if args.baseline:
        run_name = run_name if run_name != "improved" else "baseline"
        t["use_uncertainty"] = False
        t["use_boundary"]    = False
        t["use_scheduler"]   = False
    elif args.no_uncertainty:
        run_name = run_name if run_name != "improved" else "no_uncertainty"
        t["use_uncertainty"] = False
        t["use_boundary"]    = True
        t["use_scheduler"]   = True
    else:
        t["use_uncertainty"] = True
        t["use_boundary"]    = True
        t["use_scheduler"]   = True

    # Separate checkpoint and output dirs per run so they never overwrite each other
    t["checkpoint_dir"] = f"checkpoints/finetune_{run_name}"

    davis_root = cfg["data"]["davis_root"]
    ckpt_dir = Path(t["checkpoint_dir"])
    output_dir = Path(cfg["evaluation"]["output_dir"]) / f"finetune_{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SAMet — Fine-Tuning Pipeline")
    print(f"  Dataset : DAVIS 2017")
    print(f"  Output  : {output_dir}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Phase 1: Training
    # -----------------------------------------------------------------------
    if not args.eval_only:
        print("\n[PHASE 1] Fine-tuning SAM 2 ...")
        train(cfg)

        # Pick latest checkpoint
        checkpoints = sorted(ckpt_dir.glob("sam2_finetune_epoch*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        latest_ckpt = str(checkpoints[-1])
        print(f"\nUsing checkpoint: {latest_ckpt}")
    else:
        latest_ckpt = args.checkpoint
        if latest_ckpt is None:
            checkpoints = sorted(ckpt_dir.glob("sam2_finetune_epoch*.pt"))
            if not checkpoints:
                raise FileNotFoundError(
                    f"No checkpoint found. Provide --checkpoint or run training first."
                )
            latest_ckpt = str(checkpoints[-1])
        print(f"\n[EVAL-ONLY] Using checkpoint: {latest_ckpt}")

    # -----------------------------------------------------------------------
    # Phase 2: Zero-shot baseline (for comparison)
    # -----------------------------------------------------------------------
    zs_results = None
    if args.compare_zero_shot:
        print("\n[PHASE 2] Running zero-shot baseline for comparison ...")
        zs_result_path = Path(cfg["evaluation"]["output_dir"]) / "zero_shot_mask" / "results_summary.json"

        if zs_result_path.exists():
            print(f"  Loading cached zero-shot results from {zs_result_path}")
            with open(zs_result_path) as f:
                zs_summary = json.load(f)
            # Need full results for plotting — re-run if full results missing
            zs_csv = zs_result_path.parent / "results.csv"
            if not zs_csv.exists():
                args.compare_zero_shot = False
            else:
                print(f"  Zero-shot mean J&F: {zs_summary['mean_JF']:.4f}")
        else:
            print("  No cached results found — running zero-shot inference ...")
            zs_predictor = load_sam2_video_predictor(cfg)
            zs_preds = run_zero_shot_on_dataset(
                zs_predictor, davis_root, split="val", prompt_type="mask", cfg=cfg
            )
            gt = get_gt_masks_for_evaluation(davis_root, split="val")
            zs_results = evaluate_dataset(zs_preds, gt, compute_f=True)
            print_results(zs_results)
            del zs_predictor

    # -----------------------------------------------------------------------
    # Phase 3: Fine-tuned evaluation
    # -----------------------------------------------------------------------
    print("\n[PHASE 3] Evaluating fine-tuned model on DAVIS 2017 val ...")
    ft_predictor = load_finetuned_predictor(cfg, latest_ckpt)
    ft_preds = run_zero_shot_on_dataset(
        ft_predictor, davis_root, split="val", prompt_type="mask", cfg=cfg
    )
    gt = get_gt_masks_for_evaluation(davis_root, split="val")
    ft_results = evaluate_dataset(ft_preds, gt, compute_f=True)

    print("\n--- Fine-tuned Results ---")
    print_results(ft_results)

    # Save results
    save_results_csv(ft_results, str(output_dir / "results_finetuned.csv"))
    with open(output_dir / "results_finetuned_summary.json", "w") as f:
        json.dump(
            {
                "mean_J": ft_results["mean_J"],
                "mean_F": ft_results["mean_F"],
                "mean_JF": ft_results["mean_JF"],
                "checkpoint": str(latest_ckpt),
            },
            f, indent=2,
        )

    # Save predicted masks
    save_predictions(ft_preds, str(output_dir / "masks"))

    # -----------------------------------------------------------------------
    # Phase 4: Visualizations
    # -----------------------------------------------------------------------
    print("\n[PHASE 4] Generating visualizations ...")
    viz_dir = output_dir / "visualizations"
    dataset = DAVISDataset(davis_root, split="val")

    for seq in dataset.sequences[:cfg["evaluation"].get("num_viz_sequences") or 10]:
        frames = [seq.load_frame(i) for i in range(len(seq))]
        gt_masks = {
            oid: [seq.load_binary_mask(i, oid) for i in range(len(seq))]
            for oid in seq.object_ids
        }
        pred = ft_preds.get(seq.name, {})
        if not pred:
            continue
        save_sequence_overlay(frames, pred, str(viz_dir), seq.name,
                              gt_masks=gt_masks, step=5)

    # Bar chart
    plot_jf_bar_chart(ft_results, str(output_dir / "jf_bar_chart_finetuned.png"))

    # Training curves
    tb_dir = str(ckpt_dir / "tensorboard")
    plot_training_curves(tb_dir, str(output_dir / "training_curves.png"))

    # Zero-shot vs fine-tuned comparison
    if zs_results is not None:
        plot_zero_shot_vs_finetuned(
            zs_results, ft_results,
            str(output_dir / "comparison_zs_vs_ft.png")
        )

    print(f"\n{'='*60}")
    print(f"Fine-tuned mean J&F: {ft_results['mean_JF']:.4f}")
    if zs_results:
        delta = ft_results["mean_JF"] - zs_results["mean_JF"]
        print(f"Zero-shot mean J&F:  {zs_results['mean_JF']:.4f}")
        print(f"Improvement:         {delta:+.4f}")
    print(f"{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
