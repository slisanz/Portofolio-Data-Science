from __future__ import annotations

import streamlit as st

st.title("Methodology")

st.markdown(
    """
### Source

Arcep's quarterly observatory of the fixed broadband and very-high-speed market,
specifically the building-level resource for Troyes Champagne Métropole
(`e-3.csv`). The dataset is published under an open licence at
[data.europa.eu](https://data.europa.eu/data/datasets/66cc737273555682b1e9b051?locale=en).

### Cleaning

- File is encoded **cp1252**, not UTF-8. Reading it as UTF-8 mangles `é`, `è`, etc.
- Empty strings are converted to NA before any string operation.
- `code_poste` is normalised to a 5-character zero-padded string.
- Coordinates outside metropolitan-France bounds are dropped.
- The target `is_lagging` is `True` when `imb_etat != 'deploye'` — the
  building-level service state. The PM-level state (`pm_etat`) is essentially
  uniform in this snapshot so it carries no signal; the actual rollout
  variation lives at the building level.

### Coordinates

Despite the column names `x` and `y`, the raw values are already WGS-84 lon/lat,
not Lambert-93 metres. `geo.detect_crs` checks the value range and short-circuits
when the input is already in degrees.

### Equity index

A weighted sum of three sub-indicators, each in [0, 1] and oriented so higher = better:

| Sub-indicator  | Definition                                                  | Default weight |
| -------------- | ----------------------------------------------------------- | -------------- |
| coverage       | raw share of buildings with `imb_etat == 'deploye'`         | 0.60           |
| pm_load        | 1 − rank-percentile of buildings-per-PM                     | 0.20           |
| coll_lag       | share of *collective* buildings (≥2 dwellings) deployed     | 0.20           |

**Why only three, not four.** We considered an operator-competition dimension
(1 − HHI of `code_l331`) and a recency dimension (deployment-age based), but
both collapsed to constants on this snapshot — Troyes is single-operator per
RIP zone (HHI = 1 everywhere) and `date_completude` is uniformly null. Adding
zero-variance dimensions to the composite would dilute the meaningful signal
without adding information.

`coverage` uses the raw 0–1 value (no min-max), so a commune with 80% coverage
gets 0.8 even if it's the worst in the metropolis. `pm_load` is rank-pct'd
because the raw `buildings_per_pm` distribution is heavy-tailed and partly a
ruralness proxy.

`coll_lag` separates out collective dwellings because in this dataset
collective buildings lag 2-3× more than detached homes. A commune can have
high overall coverage but neglect its multi-dwelling stock; this sub-indicator
surfaces that pattern.

### Lagging-deployment model

LightGBM (or sklearn HistGB fallback) wrapped in `CalibratedClassifierCV` with
isotonic regression. Group-aware 5-fold CV over `code_insee` to avoid leakage
across communes. Reported metrics: ROC-AUC, average precision, Brier score on a
20% holdout.

### What this does *not* say

- The dataset is one quarterly snapshot. We can't measure *speed* of rollout.
- "Lagging" lumps two distinct phenomena (missing completion date vs.
  not-yet-deployed PM). They likely have different generative processes.
- No socioeconomic data is joined yet, so the equity index measures
  *infrastructural* equity, not *outcome* equity for residents.
- The clustering and uplift notebooks are exploratory; do not treat them as
  causal evidence.
"""
)
