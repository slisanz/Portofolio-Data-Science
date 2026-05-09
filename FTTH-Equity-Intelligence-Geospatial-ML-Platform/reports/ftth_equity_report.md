# FTTH Equity in Troyes Champagne Métropole

A short writeup of the analysis. This is the version I'd hand to a non-technical
reader; the notebooks have the full work.

## What I looked at

Arcep publishes a quarterly building-level snapshot of fibre-to-the-home
deployments. The `e-3.csv` file covers ~57k buildings in the Troyes Champagne
Métropole. For each building I have its position, its building category
(detached / small collective / large collective), the mutualisation point (PM)
it sits behind, the operator code, and (when known) the date the deployment
became complete.

I treated rollout as a digital-equity question: which communes are best served,
which are lagging, and can we predict at the building level which addresses are
most likely to still be incomplete.

## Headline numbers

Once notebook 03 has run, `reports/figures/headline.json` carries:

- `n_buildings`: total buildings after coordinate-bounds filtering
- `n_communes`: distinct INSEE codes
- `n_pm`: distinct mutualisation points
- `lagging_share`: fraction of buildings flagged `is_lagging`

A reader of the dashboard can copy these straight into the *Overview* page.

## Equity index

A composite per-commune score with three sub-indicators:

- **coverage**: raw share of buildings with `imb_etat == 'deploye'`
- **pm_load**: rank-percentile of buildings-per-PM, inverted (lower = more
  contention per access point = lower score)
- **coll_lag**: share of *collective* buildings (≥2 dwellings) that are deployed

Default weights 0.6 / 0.2 / 0.2. Coverage dominates because it's the most
direct measure. The methodology page explains how to re-weight.

We initially had two more sub-indicators (recency, operator competition) but
both collapsed to constants on the Troyes snapshot. Recency was based on
`date_completude` which is uniformly null in the data we have. Competition
(1 minus HHI of `code_l331` per commune) is essentially 0 everywhere because
the French RIP model assigns single infrastructure operators per zone. Carrying
zero-variance dimensions would dilute the meaningful signal, so we dropped
them rather than keeping them for symmetry.

The headline number a reader will look at is `coverage`. The other two
sub-indicators provide texture: `pm_load` highlights densely-served
neighbourhoods where service per access point is contention-heavy, and
`coll_lag` flags communes that have OK overall coverage but are leaving their
multi-dwelling buildings behind.

## Predicting lag

A LightGBM classifier predicts whether a given building is "lagging", defined
as `imb_etat != 'deploye'`. The base rate sits around 5 to 6%, with strong
spatial variation: some communes are essentially zero, others sit above 20%.

Group-aware 5-fold CV on `code_insee` to avoid leakage between train and
validation sets within the same commune. Probabilities are isotonic-calibrated
so the predicted P(lagging) is interpretable as a frequency.

The feature set deliberately excludes any aggregate of the target (no
`pm_share_lagging`, no `com_share_lagging`). Including those would let the
model read the answer through the PM- or commune-level mean, inflating AUC
without learning anything generalisable.

A useful sanity check: the cross-validated AUC (groups by commune) is
markedly lower than the random-holdout AUC. That's expected and informative.
Predicting *within* a known commune is much easier than predicting in a
commune the model has never seen. The gap quantifies how much of the signal
is local versus generalisable.

The model isn't a forecast; it's a flag. It tells you which buildings *look
like* the kind of building that tends to lag, conditional on the snapshot.

## What I'd add next

- Join INSEE filosofi income deciles per IRIS to turn the equity index from
  *infrastructural* to *outcome*-oriented.
- Pull commune polygons from IGN ADMIN-EXPRESS for proper choropleth maps.
- Move from a single snapshot to the multi-quarter panel and look at *speed*.

## Caveats

- One snapshot, no panel.
- The "lagging" label is a coarse proxy; treating "missing date" and "PM not
  deployed" as identical is a simplification I'd revisit with a longitudinal
  view.
- No causal claims. The uplift notebook is exploratory only.
