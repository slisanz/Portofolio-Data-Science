"""Composite equity index at the commune level.

The Troyes snapshot collapses two of the four sub-indicators we initially
considered into constants — `pm_etat` is essentially uniform (everything is
deployed at PM level) and `code_l331` is single-operator per RIP zone, so
HHI = 1 everywhere. We dropped both rather than carry dead weight.

What remains varies meaningfully across communes:

    coverage     = share of buildings with `imb_etat == 'deploye'` (raw, 0-1)
    pm_load      = 1 - rank-percentile of buildings-per-PM
    coll_lag     = 1 - lag share among collective buildings within the commune

`coverage` is the headline number any reader will look at. `pm_load` partly
proxies ruralness (rural communes carry fewer buildings per PM) but also reads
as "how concentrated is service per access point". `coll_lag` captures whether
the commune is leaving its multi-dwelling buildings behind — collectives lag
2-3x more than detached homes in this dataset, so a separate sub-indicator
makes sense.

Defaults: 0.6 / 0.2 / 0.2. Coverage dominates because it is the most direct,
honest measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


COLLECTIVE_CATEGORIES = (
    "entre 2 et 11",
    "entre 12 et 35",
    "entre 36 et 99",
    "100 et plus",
)


@dataclass(frozen=True)
class EquityWeights:
    coverage: float = 0.60
    pm_load: float = 0.20
    coll_lag: float = 0.20

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.coverage, self.pm_load, self.coll_lag], dtype=float
        )


def _rank_pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")


def commune_equity(
    df: pd.DataFrame, weights: EquityWeights = EquityWeights()
) -> pd.DataFrame:
    g = df.groupby("code_insee")
    coverage = g["imb_etat"].apply(
        lambda x: (x.astype("object") == "deploye").mean()
    )

    buildings_per_pm = g.size() / g["pm_ref"].nunique().clip(lower=1)
    pm_load = 1 - _rank_pct(buildings_per_pm)

    is_coll = df["catg_loc_imb"].astype("object").isin(COLLECTIVE_CATEGORIES)
    coll = df[is_coll].copy()
    if len(coll):
        coll_dep = coll.assign(
            _ok=(coll["imb_etat"].astype("object") == "deploye").astype(float)
        )
        coll_cov = coll_dep.groupby("code_insee")["_ok"].mean()
        coll_cov = coll_cov.reindex(coverage.index).fillna(coverage)
    else:
        coll_cov = pd.Series(1.0, index=coverage.index)

    parts = pd.DataFrame(
        {
            "coverage": coverage,
            "pm_load": pm_load,
            "coll_lag": coll_cov,
        }
    )
    w = weights.as_array()
    parts["equity_index"] = parts.values @ w
    parts = parts.join(g["nom_com"].first().rename("nom_com"))
    parts["n_buildings"] = g.size()
    return parts.sort_values("equity_index")
