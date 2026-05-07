"""Streamlit dashboard for browsing CLIP rose classification results.

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


RESULTS_DIR = Path("results")


def _image_path(name: str) -> Path:
    return RESULTS_DIR / name


def _show_image(name: str, caption: str) -> None:
    path = _image_path(name)
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {path}")


def main() -> None:
    st.set_page_config(
        page_title="CLIP Rose Results Dashboard",
        page_icon="🌹",
        layout="wide",
    )

    st.title("🌹 CLIP Rose Classification Dashboard")
    st.caption("Interactive view of experiment outputs")

    summary_path = RESULTS_DIR / "summary_metrics.csv"
    if not summary_path.exists():
        st.error("`results/summary_metrics.csv` not found.")
        st.info("Run the experiment first: `python main.py`")
        return

    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        st.error("`summary_metrics.csv` is empty.")
        return

    summary_df = summary_df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    best = summary_df.iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Strategy", str(best["strategy"]))
    c2.metric("Best Accuracy", f"{float(best['accuracy']) * 100:.1f}%")
    c3.metric("Best F1 Macro", f"{float(best['f1_macro']) * 100:.1f}%")

    st.subheader("Strategy Metrics")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Accuracy", "Confusion Matrices", "Score Analysis", "Per-Image Table"]
    )

    with tab1:
        _show_image("accuracy_by_strategy.png", "Prompt strategy accuracy")

    with tab2:
        cm_files = [
            "confusion_matrix_simple.png",
            "confusion_matrix_descriptive.png",
            "confusion_matrix_scientific.png",
            "confusion_matrix_contextual.png",
            "confusion_matrix_ensemble.png",
        ]
        cols = st.columns(2)
        for idx, filename in enumerate(cm_files):
            with cols[idx % 2]:
                _show_image(filename, filename.replace(".png", "").replace("_", " "))

    with tab3:
        left, right = st.columns(2)
        with left:
            _show_image("score_heatmap.png", "Mean correct-class score heatmap")
        with right:
            _show_image("score_distribution.png", "Correct-class score distribution")

    with tab4:
        per_image_path = RESULTS_DIR / "per_image_scores.csv"
        if per_image_path.exists():
            per_image_df = pd.read_csv(per_image_path)
            st.dataframe(per_image_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download per_image_scores.csv",
                data=per_image_path.read_bytes(),
                file_name="per_image_scores.csv",
                mime="text/csv",
            )
        else:
            st.warning("`results/per_image_scores.csv` not found.")


if __name__ == "__main__":
    main()
