from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ftth_equity import config, io  # noqa: E402

st.title("Overview")

clean_path = config.INTERIM / "buildings_clean.parquet"
if not io.artifact_exists(clean_path):
    st.warning("Run `notebooks/02_cleaning_and_geocoding.ipynb` first.")
    st.stop()

df = io.load_parquet(clean_path)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Buildings", f"{len(df):,}")
c2.metric("Communes", df["code_insee"].nunique())
c3.metric("PMs", df["pm_ref"].nunique())
lag_pct = df["is_lagging"].mean() * 100 if "is_lagging" in df else 0
c4.metric("Lagging share", f"{lag_pct:.1f}%")

st.subheader("Building category mix")
mix = df["catg_loc_imb"].value_counts(dropna=False).rename_axis("category").reset_index(name="n")
st.bar_chart(mix.set_index("category"))

st.subheader("Deployment state by PM")
pm = df["pm_etat"].value_counts(dropna=False).rename_axis("state").reset_index(name="n")
st.bar_chart(pm.set_index("state"))

if "date_completude" in df:
    st.subheader("Completion dates")
    s = pd.to_datetime(df["date_completude"], errors="coerce").dropna()
    if len(s):
        st.line_chart(s.dt.to_period("M").value_counts().sort_index())
    else:
        st.info("No completion dates in the cleaned set.")
