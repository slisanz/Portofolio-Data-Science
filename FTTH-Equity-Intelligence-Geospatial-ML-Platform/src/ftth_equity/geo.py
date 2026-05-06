from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

try:
    import h3
except ImportError:  # h3 v4 has a different API; we pin v3 in requirements
    h3 = None


def detect_crs(df: pd.DataFrame) -> str:
    """Cheap heuristic: degrees vs metres."""
    if df["x"].abs().max() < 180 and df["y"].abs().max() < 90:
        return config.CRS_WGS84
    return config.CRS_LAMBERT93


def to_wgs84(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    crs = detect_crs(out)
    if crs == config.CRS_WGS84:
        out["lon"] = out["x"]
        out["lat"] = out["y"]
        return out
    from pyproj import Transformer

    tf = Transformer.from_crs(config.CRS_LAMBERT93, config.CRS_WGS84, always_xy=True)
    lon, lat = tf.transform(out["x"].to_numpy(), out["y"].to_numpy())
    out["lon"] = lon
    out["lat"] = lat
    return out


def add_h3(df: pd.DataFrame, resolution: int = config.H3_RESOLUTION) -> pd.DataFrame:
    if h3 is None:
        raise RuntimeError("h3 not installed")
    out = df.copy()
    out["h3"] = [
        h3.geo_to_h3(lat, lon, resolution)
        for lat, lon in zip(out["lat"].to_numpy(), out["lon"].to_numpy())
    ]
    return out


def commune_centroids(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("code_insee").agg(
        nom_com=("nom_com", "first"),
        lon=("lon", "mean"),
        lat=("lat", "mean"),
        n_buildings=("imb_id", "count"),
    )
    return g.reset_index()


def bbox(df: pd.DataFrame, pad: float = 0.01) -> tuple[float, float, float, float]:
    return (
        df["lon"].min() - pad,
        df["lat"].min() - pad,
        df["lon"].max() + pad,
        df["lat"].max() + pad,
    )


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * r * np.arcsin(np.sqrt(a))
