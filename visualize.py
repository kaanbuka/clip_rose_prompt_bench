"""Plot generation for the CLIP rose classification experiment.

Produces:
  1. accuracy_by_strategy.png   - strategies ranked by accuracy.
  2. confusion_matrix_<s>.png   - one row-normalised heatmap per strategy.
  3. score_heatmap.png          - mean correct-class score per (strategy, class).
  4. score_distribution.png     - violin plot of correct-class scores.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import RESULTS_DIR, ROSE_KEYS
from prompt_strategies import STRATEGY_DESCRIPTIONS

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = {
    "simple":      "#7F77DD",
    "descriptive": "#1D9E75",
    "scientific":  "#EF9F27",
    "contextual":  "#D85A30",
    "ensemble":    "#D4537E",
}
ROSE_DISPLAY = {
    "rosa_canina":     "R. canina",
    "rosa_damascena":  "R. damascena",
    "rosa_multiflora": "R. multiflora",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_accuracy(summary_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [PALETTE.get(s, "#888") for s in summary_df["strategy"]]
    bars = ax.barh(
        summary_df["strategy"], summary_df["accuracy"],
        color=colors, height=0.55, edgecolor="none"
    )
    ax.bar_label(bars, fmt="%.1%", padding=4, fontsize=10)
    ax.set_xlabel("Zero-Shot Accuracy")
    ax.set_title("Accuracy by Prompt Strategy\n(CLIP ViT-B/32 · 3 rose classes)", pad=12)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right", "left"]].set_visible(False)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(summary_df)))
    ax2.set_yticklabels(
        [STRATEGY_DESCRIPTIONS.get(s, "") for s in summary_df["strategy"]],
        fontsize=8.5, color="#888"
    )
    ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)

    _save(fig, out_dir / "accuracy_by_strategy.png")


def plot_confusion_matrices(results: dict, out_dir: Path) -> None:
    strategies = results["strategies"]
    predictions = results["predictions"]
    true_labels = results["true_labels"]
    labels = [ROSE_DISPLAY[k] for k in ROSE_KEYS]

    for j, strat_name in enumerate(strategies):
        preds = predictions[:, j]
        cm = confusion_matrix(true_labels, preds, labels=list(range(len(ROSE_KEYS))))
        # Normalize row-wise for color intensity while annotating raw counts.
        with np.errstate(invalid="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm_norm, annot=cm, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=ax, linewidths=0.5, linecolor="#eee",
            vmin=0, vmax=1, cbar_kws={"format": "%.0%"}
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix - {strat_name}\n(color = row-normalized)", pad=8)
        _save(fig, out_dir / f"confusion_matrix_{strat_name}.png")


def plot_score_heatmap(results: dict, out_dir: Path) -> None:
    """Heatmap of mean softmax score on the correct class, per (strategy, class)."""
    strategies = results["strategies"]
    scores = results["scores"]          # (N, S, C)
    true_labels = results["true_labels"]

    data = np.zeros((len(strategies), len(ROSE_KEYS)))
    for j in range(len(strategies)):
        for c, _ in enumerate(ROSE_KEYS):
            mask = true_labels == c
            if mask.any():
                data[j, c] = scores[mask, j, c].mean()

    df = pd.DataFrame(
        data,
        index=strategies,
        columns=[ROSE_DISPLAY[k] for k in ROSE_KEYS],
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        df, annot=True, fmt=".2f", cmap="YlOrRd",
        ax=ax, linewidths=0.5, linecolor="#eee",
        vmin=0, vmax=1, cbar_kws={"label": "Mean softmax score"}
    )
    ax.set_title("Mean correct-class score\n(strategy × rose type)", pad=10)
    ax.set_ylabel("Prompt strategy")
    _save(fig, out_dir / "score_heatmap.png")


def plot_score_distribution(results: dict, out_dir: Path) -> None:
    """Violin plot of correct-class softmax scores per strategy."""
    strategies = results["strategies"]
    scores = results["scores"]
    true_labels = results["true_labels"]

    rows = []
    for j, strat_name in enumerate(strategies):
        for i, true_c in enumerate(true_labels):
            rows.append({
                "strategy": strat_name,
                "correct_class_score": float(scores[i, j, true_c]),
            })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = strategies
    sns.violinplot(
        data=df, x="strategy", y="correct_class_score",
        order=order, palette=PALETTE, inner="quart",
        ax=ax, cut=0, linewidth=0.8
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Prompt strategy")
    ax.set_ylabel("Softmax score (correct class)")
    ax.set_title("Distribution of correct-class scores across prompt strategies", pad=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_dir / "score_distribution.png")


def generate_all_plots(results: dict, summary_df: pd.DataFrame, out_dir: str | Path = RESULTS_DIR) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating plots...")
    plot_accuracy(summary_df, out_dir)
    plot_confusion_matrices(results, out_dir)
    plot_score_heatmap(results, out_dir)
    plot_score_distribution(results, out_dir)
    print("All plots saved.")
