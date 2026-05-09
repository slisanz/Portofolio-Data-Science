from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ftth_equity import config, io, viz  # noqa: E402

st.title("Map Explorer")

clean_path = config.INTERIM / "buildings_clean.parquet"
if not io.artifact_exists(clean_path):
    st.warning("Run `notebooks/02_cleaning_and_geocoding.ipynb` first.")
    st.stop()

df = io.load_parquet(clean_path)
if "lon" not in df.columns:
    df["lon"] = df["x"]
    df["lat"] = df["y"]

mode = st.radio("Layer", ["Sampled points", "Hex aggregation"], horizontal=True)
sample = st.slider("Point sample size", 500, 20000, 5000, step=500)

filt = df.copy()
communes = sorted(filt["nom_com"].dropna().unique().tolist())
choice = st.multiselect("Communes", communes, default=[])
if choice:
    filt = filt[filt["nom_com"].isin(choice)]

if mode == "Sampled points":
    try:
        from streamlit_folium import st_folium

        m = viz.folium_points(filt, sample=sample)
        st_folium(m, width=900, height=600)
    except ImportError:
        st.warning("streamlit-folium not installed. Run `pip install streamlit-folium`.")
else:
    try:
        deck = viz.pydeck_hex(filt)
        st.pydeck_chart(deck)
    except Exception as e:
        st.error(f"pydeck failed: {e}. Try the points layer.")
