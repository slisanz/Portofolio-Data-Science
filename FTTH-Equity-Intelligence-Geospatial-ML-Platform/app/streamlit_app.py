from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ftth_equity import config, io, models  # noqa: E402

st.set_page_config(
    page_title="FTTH Equity Intelligence",
    layout="wide",
)

st.title("FTTH Equity Intelligence")
st.caption(
    "Building-level fibre rollout across Troyes Champagne Métropole. "
    "Source: Arcep open observatory (e-3.csv)."
)

st.markdown(
    """
This dashboard reads artifacts produced by the notebooks in `notebooks/`.
If a page tells you something is missing, run the notebook it points at and reload.
"""
)

st.subheader("Pages")
st.markdown(
    """
| Page | What it shows |
| :--- | :--- |
| **Overview** | Headline numbers and data freshness. |
| **Map Explorer** | Building-level map with a lagging-deployment overlay. |
| **Equity Index** | Commune ranking and sub-indicator breakdown. |
| **Deployment Predictor** | A calibrated LightGBM "what-if" probability for a single building. |
| **Methodology** | How each metric is built, and what it does **not** say. |
"""
)

st.subheader("Pipeline status")

clean_path = config.INTERIM / "buildings_clean.parquet"
features_path = config.PROCESSED / "buildings_features.parquet"
equity_path = config.PROCESSED / "commune_equity.parquet"
model_path = config.MODELS / "lag_classifier.joblib"


def _row_count(path):
    if not io.artifact_exists(path):
        return None
    try:
        return len(io.load_parquet(path))
    except Exception:
        return None


def _fmt_count(n, suffix):
    return "not built" if n is None else f"{n:,} {suffix}"


def _holdout_auc(path):
    if not io.artifact_exists(path):
        return None
    try:
        return models.load(path)["metrics"]["holdout_auc"]
    except Exception:
        return None


buildings_n = _row_count(clean_path)
features_n = _row_count(features_path)
equity_n = _row_count(equity_path)
auc = _holdout_auc(model_path)

cols = st.columns(4)
cols[0].metric("Cleaned buildings", _fmt_count(buildings_n, "rows"))
cols[1].metric("Feature table", _fmt_count(features_n, "rows"))
cols[2].metric("Equity index", _fmt_count(equity_n, "communes"))
cols[3].metric("Lag model AUC", "not built" if auc is None else f"{auc:.3f}")

if buildings_n is None:
    st.warning(
        "No cleaned data yet. Start by running `notebooks/02_cleaning_and_geocoding.ipynb`."
    )
else:
    st.subheader("Sample of cleaned buildings")
    df = io.load_parquet(clean_path)
    st.dataframe(df.head(20), use_container_width=True)
