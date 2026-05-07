"""CLI entry point for the CLIP zero-shot rose classification experiment.

Usage:
    python main.py
    python main.py --image_dir data/sample_images --results_dir results
    python main.py --model ViT-L/14
"""

import argparse
import sys
from pathlib import Path

from clip_classifier import CLIPRoseClassifier
from evaluate import compute_metrics, save_reports
from visualize import generate_all_plots
from config import IMAGE_DIR, RESULTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLIP zero-shot rose classification prompt strategy comparison"
    )
    parser.add_argument(
        "--image_dir", default=IMAGE_DIR,
        help=f"Root directory containing rose class subdirectories (default: {IMAGE_DIR})"
    )
    parser.add_argument(
        "--results_dir", default=RESULTS_DIR,
        help=f"Directory to save results and plots (default: {RESULTS_DIR})"
    )
    parser.add_argument(
        "--model", default="ViT-B/32",
        help="CLIP model variant (default: ViT-B/32)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"[ERROR] Image directory not found: {image_dir}")
        print("Please create subdirectories for each rose class:")
        print("  data/sample_images/rosa_canina/")
        print("  data/sample_images/rosa_damascena/")
        print("  data/sample_images/rosa_multiflora/")
        sys.exit(1)

    print("=" * 60)
    print("  CLIP Zero-Shot Rose Classification")
    print("  Prompt Strategy Comparison Experiment")
    print("=" * 60)

    classifier = CLIPRoseClassifier(model_name=args.model)
    results = classifier.run_experiment(image_dir)

    summary_df = compute_metrics(results)
    save_reports(results, out_dir=args.results_dir)

    generate_all_plots(results, summary_df, out_dir=args.results_dir)

    print("\n" + "=" * 60)
    print("  Experiment complete!")
    print(f"  Results saved to: {Path(args.results_dir).resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
