from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the Arcep building-level CSV with the right encoding and dtypes."""
    path = path or config.RAW_CSV
    df = pd.read_csv(
        path,
        encoding=config.RAW_ENCODING,
        index_col=0,
        dtype={
            "imb_id": "string",
            "code_insee": "string",
            "pm_ref": "string",
            "code_l331": "string",
            "type_imb": "string",
            "imb_etat": "string",
            "pm_etat": "string",
            "catg_loc_imb": "string",
            "nom_com": "string",
            "type_voie": "string",
            "nom_voie": "string",
            "batiment": "string",
            "cp_no_voie": "string",
            "geom_mod": "string",
        },
        low_memory=False,
    )
    df.index.name = "row_id"
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    return path


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def artifact_exists(path: Path) -> bool:
    return Path(path).exists()
