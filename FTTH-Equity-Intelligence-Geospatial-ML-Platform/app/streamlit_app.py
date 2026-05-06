from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ftth_equity import config, io  # noqa: E402

st.set_page_config(
    page_title="FTTH Equity Intelligence",
    page_icon=":satellite:",
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

**Pages**

- *Overview* — headline numbers and data freshness
- *Map Explorer* — building-level map with lagging-deployment overlay
- *Equity Index* — commune ranking + sub-indicator breakdown
- *Deployment Predictor* — ML "what-if" using a calibrated LightGBM model
- *Methodology* — how each metric is built and what it does **not** say
"""
)

clean_path = config.INTERIM / "buildings_clean.parquet"
features_path = config.PROCESSED / "buildings_features.parquet"
equity_path = config.PROCESSED / "commune_equity.parquet"
model_path = config.MODELS / "lag_classifier.joblib"

cols = st.columns(4)
cols[0].metric("Cleaned buildings", "—" if not io.artifact_exists(clean_path) else "ok")
cols[1].metric("Feature table", "—" if not io.artifact_exists(features_path) else "ok")
cols[2].metric("Equity index", "—" if not io.artifact_exists(equity_path) else "ok")
cols[3].metric("Lag model", "—" if not io.artifact_exists(model_path) else "ok")

if not io.artifact_exists(clean_path):
    st.warning(
        "No cleaned data yet. Start by running `notebooks/02_cleaning_and_geocoding.ipynb`."
    )
else:
    df = io.load_parquet(clean_path)
    st.success(f"{len(df):,} cleaned buildings loaded.")
    st.dataframe(df.head(20), use_container_width=True)
