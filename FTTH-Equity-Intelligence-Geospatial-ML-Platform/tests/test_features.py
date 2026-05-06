import pandas as pd

from ftth_equity import features


def _df():
    return pd.DataFrame(
        {
            "imb_id": list("abcdef"),
            "pm_ref": ["P1", "P1", "P2", "P2", "P2", "P3"],
            "code_insee": ["10001", "10001", "10001", "10002", "10002", "10002"],
            "code_l331": ["FRTE", "FRTE", "ORANGE", "ORANGE", "FRTE", "FRTE"],
            "catg_loc_imb": [
                "individuel",
                "individuel",
                "entre 2 et 11",
                "individuel",
                "100 et plus",
                "individuel",
            ],
            "batiment": [None, "A", None, "B", None, "C"],
            "code_poste": ["10000", None, "10000", "10100", "10100", "10100"],
            "num_voie": ["1", "2", None, "5", "7", "9"],
            "is_lagging": [0, 1, 0, 1, 1, 0],
        }
    )


def test_pm_features_share_individuel_per_pm():
    out = features.pm_features(_df())
    p1 = out[out["pm_ref"] == "P1"]["pm_share_individuel"].iloc[0]
    p2 = out[out["pm_ref"] == "P2"]["pm_share_individuel"].iloc[0]
    assert p1 == 1.0
    assert abs(p2 - 1 / 3) < 1e-9


def test_pm_features_do_not_leak_target():
    out = features.pm_features(_df())
    leaky = [c for c in out.columns if "lagging" in c.lower()]
    assert leaky == ["is_lagging"], f"unexpected leaky aggregates: {leaky}"


def test_commune_buildings_per_pm():
    out = features.commune_features(_df())
    c1 = out[out["code_insee"] == "10001"]["com_buildings_per_pm"].iloc[0]
    # commune 10001 has 3 buildings across 2 PMs -> 1.5
    assert abs(c1 - 1.5) < 1e-9


def test_operator_hhi_in_unit_interval():
    out = features.operator_concentration(_df())
    assert (out["com_hhi"] >= 0).all()
    assert (out["com_hhi"] <= 1).all()


def test_operator_hhi_handles_nan_codes_consistently():
    df = _df()
    df.loc[0, "code_l331"] = None
    out = features.operator_concentration(df)
    # commune 10001: NaN, FRTE, ORANGE -> shares 1/3, 1/3, 1/3 -> HHI ~ 0.333
    h = out[out["code_insee"] == "10001"]["com_hhi"].iloc[0]
    assert abs(h - (3 * (1 / 3) ** 2)) < 1e-9


def test_full_feature_table_has_all_numeric_cols():
    out = features.build_feature_table(_df())
    for c in features.numeric_feature_columns():
        assert c in out.columns, f"missing {c}"


def test_numeric_feature_columns_excludes_target_aggregates():
    cols = features.numeric_feature_columns()
    for c in cols:
        assert "lagging" not in c.lower(), f"target aggregate in features: {c}"
