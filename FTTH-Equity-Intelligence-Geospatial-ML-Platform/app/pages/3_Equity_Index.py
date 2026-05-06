from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ftth_equity import config, io, viz  # noqa: E402

st.title("Equity Index")

equity_path = config.PROCESSED / "commune_equity.parquet"
if not io.artifact_exists(equity_path):
    st.warning("Run `notebooks/05_equity_index.ipynb` first.")
    st.stop()

eq = io.load_parquet(equity_path)

st.write(
    "Composite score per commune. Higher = more equitable. "
    "Sub-indicators: coverage, recency, density, competition."
)

c1, c2 = st.columns(2)
c1.metric("Communes scored", len(eq))
c2.metric("Median equity", f"{eq['equity_index'].median():.2f}")

top_n = st.slider("Bottom-N to show", 5, 40, 20)
fig = viz.commune_bar(eq, top=top_n)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Full table")
st.dataframe(eq.sort_values("equity_index"), use_container_width=True)
