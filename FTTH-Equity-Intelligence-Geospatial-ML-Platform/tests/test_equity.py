import pandas as pd

from ftth_equity import equity


def _df():
    return pd.DataFrame(
        {
            "code_insee": ["10001"] * 4 + ["10002"] * 4,
            "nom_com": ["A"] * 4 + ["B"] * 4,
            "pm_ref": ["P1", "P1", "P2", "P2", "P3", "P3", "P3", "P3"],
            "imb_etat": [
                "deploye", "deploye", "deploye", "deploye",
                "cible", "deploye", "cible", "cible",
            ],
            "catg_loc_imb": [
                "individuel", "entre 2 et 11", "individuel", "entre 2 et 11",
                "individuel", "entre 2 et 11", "individuel", "entre 2 et 11",
            ],
            "imb_id": list("abcdefgh"),
        }
    )


def test_commune_equity_returns_one_row_per_commune():
    out = equity.commune_equity(_df())
    assert set(out.index) == {"10001", "10002"}


def test_subindicators_in_unit_interval():
    out = equity.commune_equity(_df())
    for c in ["coverage", "pm_load", "coll_lag"]:
        assert (out[c] >= 0).all() and (out[c] <= 1).all()


def test_equity_index_higher_for_better_served_commune():
    out = equity.commune_equity(_df())
    assert out.loc["10001", "equity_index"] > out.loc["10002", "equity_index"]


def test_coverage_is_raw_share_of_imb_deploye():
    out = equity.commune_equity(_df())
    assert abs(out.loc["10001", "coverage"] - 1.0) < 1e-9
    assert abs(out.loc["10002", "coverage"] - 0.25) < 1e-9


def test_weights_sum_used_correctly():
    w = equity.EquityWeights(1 / 3, 1 / 3, 1 / 3)
    out = equity.commune_equity(_df(), weights=w)
    expected = out[["coverage", "pm_load", "coll_lag"]].mean(axis=1)
    assert (out["equity_index"] - expected).abs().max() < 1e-9


def test_collective_subindicator_isolates_collective_buildings():
    out = equity.commune_equity(_df())
    # commune 10002 collective buildings: entre-2-11 indices 5,7 -> deploye, cible -> 0.5
    assert abs(out.loc["10002", "coll_lag"] - 0.5) < 1e-9
