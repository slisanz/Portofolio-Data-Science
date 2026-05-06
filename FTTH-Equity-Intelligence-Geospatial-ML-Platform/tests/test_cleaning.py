import pandas as pd

from ftth_equity import cleaning


def _toy():
    return pd.DataFrame(
        {
            "x": [4.08, 4.06, 200.0],
            "y": [48.25, 48.29, 48.0],
            "imb_id": ["A", "B", "C"],
            "code_poste": ["10000.0", "10450", None],
            "date_completude": ["2023-06-01", None, "2024-01-15"],
            "date_completude_manquante": ["false", "true", "False"],
            "pm_etat": ["deploye", "deploye", "deploye"],
            "imb_etat": ["deploye", "cible", "deploye"],
            "catg_loc_imb": ["individuel", "weird", "entre 2 et 11"],
            "nom_com": ["  Troyes ", "Bréviandes", None],
        }
    )


def test_drop_invalid_coords_removes_out_of_france_points():
    out = cleaning.drop_invalid_coords(_toy())
    assert len(out) == 2
    assert out["x"].max() < 10


def test_postcode_zero_padded_string():
    out = cleaning.coerce_postcode(_toy())
    assert out.loc[0, "code_poste"] == "10000"
    assert out.loc[1, "code_poste"] == "10450"


def test_target_marks_buildings_not_deployed_at_imb_level():
    df = cleaning.add_target(_toy())
    # imb_etat: deploye, cible, deploye -> 0, 1, 0
    assert df["is_lagging"].tolist() == [0, 1, 0]


def test_clean_pipeline_runs_and_collapses_categories():
    df = cleaning.clean(_toy())
    assert df["catg_loc_imb"].isna().sum() >= 1
    assert "is_lagging" in df.columns
