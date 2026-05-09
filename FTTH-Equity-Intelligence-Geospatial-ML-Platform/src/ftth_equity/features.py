from __future__ import annotations

import numpy as np
import pandas as pd


def building_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["has_building_label"] = out["batiment"].notna().astype("int8")
    out["has_postcode"] = out["code_poste"].notna().astype("int8")
    out["num_voie_filled"] = pd.to_numeric(out["num_voie"], errors="coerce").fillna(-1)
    cat = out["catg_loc_imb"].astype("object")
    out["is_individuel"] = (cat == "individuel").fillna(False).astype("int8")
    out["is_collective_large"] = (
        cat.isin(["entre 36 et 99", "100 et plus"]).fillna(False).astype("int8")
    )
    return out


def pm_features(df: pd.DataFrame) -> pd.DataFrame:
    """PM-level aggregates. Excludes any aggregate of `is_lagging` to avoid leakage."""
    work = df.assign(
        _is_indiv=df["is_individuel"]
        if "is_individuel" in df.columns
        else (df["catg_loc_imb"].astype("object") == "individuel").fillna(False).astype("int8"),
        _is_large=df["is_collective_large"]
        if "is_collective_large" in df.columns
        else df["catg_loc_imb"]
        .astype("object")
        .isin(["entre 36 et 99", "100 et plus"])
        .fillna(False)
        .astype("int8"),
    )
    g = work.groupby("pm_ref")
    pm = pd.DataFrame(
        {
            "pm_n_buildings": g.size(),
            "pm_n_communes": g["code_insee"].nunique(),
            "pm_n_operators": g["code_l331"].nunique(),
            "pm_share_individuel": g["_is_indiv"].mean(),
            "pm_share_collective_large": g["_is_large"].mean(),
        }
    )
    return df.merge(pm, left_on="pm_ref", right_index=True, how="left")


def commune_features(df: pd.DataFrame) -> pd.DataFrame:
    """Commune-level aggregates. Same leakage-avoidance rule as `pm_features`."""
    work = df.assign(
        _is_indiv=df["is_individuel"]
        if "is_individuel" in df.columns
        else (df["catg_loc_imb"].astype("object") == "individuel").fillna(False).astype("int8"),
    )
    g = work.groupby("code_insee")
    com = pd.DataFrame(
        {
            "com_n_buildings": g.size(),
            "com_n_pm": g["pm_ref"].nunique(),
            "com_n_operators": g["code_l331"].nunique(),
            "com_share_individuel": g["_is_indiv"].mean(),
        }
    )
    com["com_buildings_per_pm"] = com["com_n_buildings"] / com["com_n_pm"].clip(lower=1)
    return df.merge(com, left_on="code_insee", right_index=True, how="left")


def operator_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """HHI of code_l331 within each commune. NaN operator codes are treated as their own bucket so shares always sum to 1."""
    op = df["code_l331"].fillna("__na__")
    counts = df.assign(_op=op).groupby(["code_insee", "_op"]).size()
    totals = counts.groupby(level=0).sum()
    shares = counts / totals
    hhi = (shares**2).groupby(level=0).sum().rename("com_hhi")
    return df.merge(hhi, left_on="code_insee", right_index=True, how="left")


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(building_features)
        .pipe(pm_features)
        .pipe(commune_features)
        .pipe(operator_concentration)
    )


def numeric_feature_columns() -> list[str]:
    """Features safe to feed the lag classifier (no aggregate of the target)."""
    return [
        "has_building_label",
        "has_postcode",
        "num_voie_filled",
        "is_individuel",
        "is_collective_large",
        "pm_n_buildings",
        "pm_n_communes",
        "pm_n_operators",
        "pm_share_individuel",
        "pm_share_collective_large",
        "com_n_buildings",
        "com_n_pm",
        "com_n_operators",
        "com_share_individuel",
        "com_buildings_per_pm",
        "com_hhi",
    ]
