from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ftth_equity import config, io, models  # noqa: E402

st.title("Deployment-lag predictor")

model_path = config.MODELS / "lag_classifier.joblib"
features_path = config.PROCESSED / "buildings_features.parquet"

if not io.artifact_exists(model_path):
    st.warning("Run `notebooks/07_ml_deployment_lag.ipynb` first.")
    st.stop()
if not io.artifact_exists(features_path):
    st.warning("Run `notebooks/06_feature_engineering.ipynb` first.")
    st.stop()


@st.cache_resource
def _load_model():
    return models.load(model_path)


@st.cache_data
def _load_features():
    return io.load_parquet(features_path)


bundle = _load_model()
df = _load_features()
feat_names = bundle["feature_names"]
metrics = bundle["metrics"]

st.write(
    f"**Holdout AUC** {metrics['holdout_auc']:.3f} · "
    f"**AP** {metrics['holdout_ap']:.3f} · "
    f"**Brier** {metrics['holdout_brier']:.3f}"
)

st.subheader("What-if for a single building")
sample_idx = st.number_input(
    "Pick a row index",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1,
)
row = df.iloc[int(sample_idx)]
inputs = {}
cols = st.columns(3)
for i, name in enumerate(feat_names):
    default = float(row[name]) if pd.notna(row[name]) else -1.0
    inputs[name] = cols[i % 3].number_input(name, value=default)

x = np.array([inputs[n] for n in feat_names], dtype=float).reshape(1, -1)
p = float(bundle["model"].predict_proba(x)[0, 1])
st.metric("P(lagging)", f"{p:.2%}")

st.caption(
    "Probability of the building being flagged as 'lagging' (PM not yet deployed "
    "or completion date missing). Calibrated isotonic; treat as a relative score."
)
