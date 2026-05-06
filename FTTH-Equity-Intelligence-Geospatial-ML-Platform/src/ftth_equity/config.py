from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

RAW_CSV = RAW / "e-3.csv"
RAW_ENCODING = "cp1252"

# the raw file's x/y columns are already WGS-84 lon/lat despite the projected-sounding names
CRS_WGS84 = "EPSG:4326"
CRS_LAMBERT93 = "EPSG:2154"

H3_RESOLUTION = 9

RANDOM_SEED = 13

COLUMNS = {
    "x": "x",
    "y": "y",
    "imb_id": "imb_id",
    "num_voie": "num_voie",
    "cp_no_voie": "cp_no_voie",
    "type_voie": "type_voie",
    "nom_voie": "nom_voie",
    "batiment": "batiment",
    "code_insee": "code_insee",
    "code_poste": "code_poste",
    "nom_com": "nom_com",
    "catg_loc_imb": "catg_loc_imb",
    "imb_etat": "imb_etat",
    "pm_ref": "pm_ref",
    "pm_etat": "pm_etat",
    "code_l331": "code_l331",
    "geom_mod": "geom_mod",
    "type_imb": "type_imb",
    "date_completude": "date_completude",
    "date_completude_manquante": "date_completude_manquante",
}

# building category codes encountered in the wild
BUILDING_CATEGORIES = [
    "individuel",
    "entre 2 et 11",
    "entre 12 et 35",
    "entre 36 et 99",
    "100 et plus",
]

# pm states
PM_STATES = ["deploye", "prevu", "en_cours", "abandonne"]


def ensure_dirs() -> None:
    for p in (INTERIM, PROCESSED, EXTERNAL, MODELS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
