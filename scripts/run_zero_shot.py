"""
Zero-shot SAM 2 evaluation on DAVIS 2017 validation set.

Usage:
    python scripts/run_zero_shot.py
    python scripts/run_zero_shot.py --config configs/config.yaml --prompt-type box
    python scripts/run_zero_shot.py --prompt-type point --no-save-viz
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import DAVISDataset, get_gt_masks_for_evaluation
from src.evaluate import evaluate_dataset, print_results, save_results_csv
from src.inference import (
    load_sam2_video_predictor,
    run_zero_shot_on_dataset,
    save_predictions,
)
from src.visualize import (
    plot_jf_bar_chart,
    save_sequence_overlay,
    save_error_heatmap,
    plot_training_curves,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot SAM 2 evaluation on DAVIS 2017")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--prompt-type", default="mask",
        choices=["mask", "box", "point"],
        help="Prompt type for first-frame initialization (default: mask)"
    )
    parser.add_argument("--split", default="val", help="DAVIS split to evaluate (default: val)")
    parser.add_argument("--save-masks", action="store_true", default=True,
                        help="Save predicted mask PNGs")
    parser.add_argument("--no-save-masks", dest="save_masks", action="store_false")
    parser.add_argument("--save-viz", action="store_true", default=True,
                        help="Save visualization images")
    parser.add_argument("--no-save-viz", dest="save_viz", action="store_false")
    parser.add_argument("--num-viz-seqs", type=int, default=None,
                        help="Number of sequences to visualize (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    davis_root = cfg["data"]["davis_root"]
    output_dir = Path(cfg["evaluation"]["output_dir"]) / f"zero_shot_{args.prompt_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  SAMet — Zero-Shot Evaluation")
    print(f"  Dataset : DAVIS 2017 ({args.split})")
    print(f"  Prompt  : {args.prompt_type}")
    print(f"  Output  : {output_dir}")
    print("=" * 60)

    # Load model
    predictor = load_sam2_video_predictor(cfg)

    # Run inference
    print("\n[1/3] Running inference...")
    predictions = run_zero_shot_on_dataset(
        predictor, davis_root, split=args.split, prompt_type=args.prompt_type, cfg=cfg
    )

    # Save masks
    if args.save_masks:
        mask_dir = output_dir / "masks"
        print(f"\n[2/3] Saving predicted masks to {mask_dir} ...")
        save_predictions(predictions, str(mask_dir))
    else:
        print("\n[2/3] Skipping mask saving.")

    # Evaluate
    print("\n[3/3] Computing J&F scores ...")
    ground_truth = get_gt_masks_for_evaluation(davis_root, split=args.split)
    results = evaluate_dataset(predictions, ground_truth, compute_f=True)

    print_results(results)

    # Save CSV
    csv_path = str(output_dir / "results.csv")
    save_results_csv(results, csv_path)

    # Save JSON
    json_path = output_dir / "results_summary.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "mean_J": results["mean_J"],
                "mean_F": results["mean_F"],
                "mean_JF": results["mean_JF"],
                "prompt_type": args.prompt_type,
                "split": args.split,
            },
            f, indent=2,
        )
    print(f"Summary saved to {json_path}")

    # Visualizations
    if args.save_viz:
        print("\nGenerating visualizations...")
        viz_dir = output_dir / "visualizations"
        dataset = DAVISDataset(davis_root, split=args.split)

        num_viz = args.num_viz_seqs or len(dataset.sequences)
        for seq in dataset.sequences[:num_viz]:
            frames = [seq.load_frame(i) for i in range(len(seq))]
            gt_masks = {
                oid: [seq.load_binary_mask(i, oid) for i in range(len(seq))]
                for oid in seq.object_ids
            }
            pred = predictions.get(seq.name, {})
            if not pred:
                continue
            save_sequence_overlay(
                frames, pred, str(viz_dir), seq.name,
                gt_masks=gt_masks, step=5
            )

        # Bar chart
        plot_jf_bar_chart(results, str(output_dir / "jf_bar_chart.png"))

        # Error heatmap for worst sequence
        worst_seq = min(
            results["per_sequence"].items(),
            key=lambda x: np.mean([v["JF"] for v in x[1].values()])
            if x[1] else 1.0,
        )
        print(f"Worst sequence: {worst_seq[0]}")

    print(f"\nFinal J&F: {results['mean_JF']:.4f}")
    print("Done.")


if __name__ == "__main__":
    import numpy as np  # needed for worst_seq fallback
    main()
