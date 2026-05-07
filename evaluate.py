"""Metric computation and report persistence for the CLIP rose experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import ROSE_KEYS, RESULTS_DIR
from prompt_strategies import STRATEGY_DESCRIPTIONS


def compute_metrics(results: dict) -> pd.DataFrame:
    """Compute per-strategy accuracy + macro precision/recall/F1.

    Args:
        results: dict returned by ``CLIPRoseClassifier.run_experiment``.

    Returns:
        DataFrame sorted by accuracy with columns
        [strategy, description, accuracy, precision_macro, recall_macro, f1_macro].
    """
    strategies = results["strategies"]
    predictions = results["predictions"]
    true_labels = results["true_labels"]

    rows = []
    for j, strat_name in enumerate(strategies):
        preds = predictions[:, j]
        acc = accuracy_score(true_labels, preds)
        report = classification_report(
            true_labels, preds,
            target_names=ROSE_KEYS,
            output_dict=True,
            zero_division=0,
        )
        rows.append({
            "strategy": strat_name,
            "description": STRATEGY_DESCRIPTIONS.get(strat_name, ""),
            "accuracy": round(acc, 4),
            "precision_macro": round(report["macro avg"]["precision"], 4),
            "recall_macro": round(report["macro avg"]["recall"], 4),
            "f1_macro": round(report["macro avg"]["f1-score"], 4),
        })

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def save_reports(results: dict, out_dir: str | Path = RESULTS_DIR) -> None:
    """Save summary CSV, per-image scores CSV, and confusion matrices."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = results["strategies"]
    predictions = results["predictions"]
    true_labels = results["true_labels"]
    scores = results["scores"]
    image_records = results["images"]

    summary = compute_metrics(results)
    summary_path = out_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary metrics saved → {summary_path}")
    print(summary.to_string(index=False))

    rows = []
    for i, (true_key, img_path) in enumerate(image_records):
        row = {
            "image": img_path.name,
            "true_class": true_key,
        }
        for j, strat_name in enumerate(strategies):
            for c, class_key in enumerate(ROSE_KEYS):
                row[f"{strat_name}_{class_key}"] = round(float(scores[i, j, c]), 4)
            row[f"{strat_name}_predicted"] = ROSE_KEYS[predictions[i, j]]
        rows.append(row)

    scores_df = pd.DataFrame(rows)
    scores_path = out_dir / "per_image_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"Per-image scores saved → {scores_path}")

    cm_path = out_dir / "confusion_matrices.txt"
    with open(cm_path, "w") as f:
        for j, strat_name in enumerate(strategies):
            preds = predictions[:, j]
            cm = confusion_matrix(true_labels, preds)
            f.write(f"\nStrategy: {strat_name}\n")
            f.write(f"{'':20s}" + "  ".join(f"{k:14s}" for k in ROSE_KEYS) + "\n")
            for r_idx, row in enumerate(cm):
                f.write(f"{ROSE_KEYS[r_idx]:20s}" + "  ".join(f"{v:14d}" for v in row) + "\n")
            f.write("\n")
    print(f"Confusion matrices saved → {cm_path}")
