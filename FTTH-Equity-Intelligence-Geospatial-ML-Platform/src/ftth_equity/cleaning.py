from __future__ import annotations

import pandas as pd

from . import config


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    str_cols = out.select_dtypes(include=["string", "object"]).columns
    for c in str_cols:
        out[c] = out[c].astype("string").str.strip()
        out.loc[out[c].eq(""), c] = pd.NA
    return out


def coerce_postcode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "code_poste" in out.columns:
        # raw is float-ish ('10000.0'); keep as 5-char zero-padded string
        out["code_poste"] = (
            pd.to_numeric(out["code_poste"], errors="coerce")
            .astype("Int64")
            .astype("string")
            .str.zfill(5)
        )
    return out


def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date_completude" in out.columns:
        out["date_completude"] = pd.to_datetime(
            out["date_completude"], errors="coerce", dayfirst=False
        )
    if "date_completude_manquante" in out.columns:
        out["date_completude_manquante"] = (
            out["date_completude_manquante"]
            .astype("string")
            .str.lower()
            .map({"true": True, "false": False, "t": True, "f": False})
            .astype("boolean")
        )
    return out


def drop_invalid_coords(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["x"].between(-5, 10) & out["y"].between(41, 52)]
    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binary target: building-level deployment is not yet 'deploye'.

    The project initially used `pm_etat` for this, but in the Troyes snapshot
    `pm_etat` is essentially a constant; only 8 of 57072 PMs are not deployed.
    The variance in the data sits at the *building* level (`imb_etat`), where
    around 5.6% of addresses are still in states like 'cible', 'signe',
    'en cours de deploiement', 'raccordable demande' or 'abandonne'.
    """
    out = df.copy()
    state = out["imb_etat"].astype("object")
    out["is_lagging"] = state.ne("deploye").fillna(True).astype("int8")
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.pipe(normalize_strings)
        .pipe(coerce_postcode)
        .pipe(coerce_dates)
        .pipe(drop_invalid_coords)
        .pipe(add_target)
    )
    # collapse rare building categories
    if "catg_loc_imb" in out.columns:
        keep = set(config.BUILDING_CATEGORIES)
        mask = out["catg_loc_imb"].isin(keep)
        out["catg_loc_imb"] = out["catg_loc_imb"].where(mask, pd.NA).astype("string")
    return out
