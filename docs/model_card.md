# Model Card — Apollo Model A

> Template follows Mitchell et al., "Model Cards for Model Reporting."
> Numbers below are from the untouched temporal test set (incident year ≥ 2019).
> Regenerate with `python -m training.train` (metrics) and `python -m training.audit`
> (fairness); see [`reports/model_A_metrics.json`](../reports/model_A_metrics.json)
> and [`reports/fairness_A.json`](../reports/fairness_A.json).

## Model details
- **Name / version:** `model_A_20260727T031353Z`
- **Date:** 2026-07-27
- **Type:** Binary classifier — calibrated probability of ≥1 fatality given an incident scenario.
- **Algorithm:** `RandomForestClassifier` (300 trees, `min_samples_leaf=4`) inside a
  `ColumnTransformer` (median-impute + scale numerics; most-frequent-impute + one-hot
  categoricals, `min_frequency=20`), wrapped in `CalibratedClassifierCV(method="isotonic", cv=3)`.
- **Training data:** GTD, combined 1970–2021 (214,666 rows), pinned SHA-256
  `1006c27e8fd5bf1936ff671aca8466bc5792a9c1f84c61a3c9e16bb3e6311bc0`. See [`data/registry.yaml`](../data/registry.yaml).
- **Features (15):** `iyear, imonth, latitude, longitude, country, region, specificity,
  vicinity, attacktype1, targtype1, targsubtype1, natlty1, weaptype1, weapsubtype1, suicide`
  — PRE_EVENT + SCENARIO only, per [`docs/prediction_time_contract.md`](prediction_time_contract.md).
  Outcome fields (`nkill`, `nwound`, `success`, …) are excluded by [`engine/leakage.py`](../engine/leakage.py)
  and cannot enter the feature set (guarded by tests).

## Intended use
See [`docs/intended_use.md`](intended_use.md). Retrospective, probabilistic decision-support.
**Not** early warning, **not** causal, **not** person-scoring.

## Out-of-scope / prohibited use
- Predicting where/when an attack will occur.
- Any action against an individual, nationality, or protected group.
- Commercial or public redistribution of raw GTD data.

## Evaluation
- **Split (temporal, no random split):** train ≤2015 (n=148,489, pos 0.478) ·
  val 2016–2018 (n=32,973, pos 0.535) · test ≥2019 (n=20,253, pos 0.590). All imputers,
  encoders, and calibration are fit inside training data only.

| Metric | Model A (test) |
|---|---|
| ROC-AUC | **0.840** |
| PR-AUC | 0.856 |
| Brier | **0.169** |
| Accuracy | 0.750 |
| F1 | 0.820 |

- **Baselines (all beaten on AUC & Brier — gate PASSED):** prevalence (AUC 0.500, Brier 0.255);
  attack-type group prevalence (AUC 0.732, Brier 0.207).
- **Calibration:** well-calibrated at the extremes (e.g. predicted 0.84 → observed 0.87;
  predicted 0.05 → observed 0.00), but **over-confident in the 0.4–0.7 mid-range** on this
  out-of-time test set (predicted ~0.55 → observed ~0.39). Treat mid-range probabilities as
  soft. This is temporal drift: calibration was fit on ≤2015 data. Full curve in the report.

## Fairness / subgroup analysis
Per-subgroup metrics on the test set (region / country / target nationality), threshold 0.5,
minimum group size 100 rows (smaller groups withheld — 4 regions, 94 countries, 110
nationalities suppressed). Full tables in [`reports/fairness_A.json`](../reports/fairness_A.json).

**By region (8 reported):**

| Region | n | base rate | TPR | FPR | ROC-AUC | calib. gap |
|---|--:|--:|--:|--:|--:|--:|
| South Asia | 8,015 | 0.70 | 0.98 | 0.53 | 0.876 | +0.00 |
| Sub-Saharan Africa | 4,374 | 0.71 | 0.97 | 0.71 | 0.753 | +0.01 |
| Middle East & N. Africa | 4,656 | 0.45 | 0.96 | 0.76 | 0.749 | **+0.22** |
| Southeast Asia | 1,422 | 0.53 | 0.93 | 0.46 | 0.821 | +0.10 |
| South America | 675 | 0.31 | 0.98 | 0.33 | 0.917 | +0.19 |
| Western Europe | 584 | 0.07 | 0.47 | 0.02 | 0.955 | +0.08 |
| North America | 297 | 0.24 | 0.90 | 0.19 | 0.923 | +0.12 |
| Eastern Europe | 120 | 0.28 | 0.53 | 0.16 | 0.854 | +0.08 |

**Key disparities:**
- **TPR ranges 0.47 (Western Europe) → 0.98 (South Asia)** — a 0.51 spread, largely driven by
  differing base rates: where fatal incidents are common the fixed 0.5 threshold catches
  nearly all of them; where they are rare (Western Europe, 7%) the model misses over half.
  Per-region **AUC is more comparable (0.75–0.96)**; discrimination is weakest in MENA and
  Sub-Saharan Africa (~0.75).
- **Calibration gap is worst in MENA (+0.22)** and South America (+0.19): the model
  systematically **over-predicts** fatality there. At country/nationality level the worst
  calibration gap is Saudi Arabia (~+0.57). Over-prediction in a lethality estimator risks
  overstating danger for specific regions/nationalities — the most decision-relevant harm here.

**Sensitive-field ablation** (`country`, `region`, `natlty1`): training the same pipeline
without these fields costs **only 0.021 ROC-AUC (0.840 → 0.819)** and 0.010 Brier
(0.169 → 0.180). The sensitive geography contributes little to overall accuracy — a
deployment prioritising fairness could drop these fields at modest cost. Retained in the
default model pending a human decision recorded here.

## Limitations
- Data ends 2021; reporting intensity and coding practice vary by era/region — GTD records
  *reported* incidents, not ground truth.
- Modest signal after correct removal of casualty fields; honest AUC (0.84) is materially
  below any leaky figure (draft 1's ~86% accuracy measured the wrong thing).
- Mid-range probabilities are over-confident out-of-time (see Calibration); the min–max
  uncertainty band the API returns reflects fold disagreement, not a formal CI.
- Subgroup TPR/FPR disparities are entangled with base-rate differences; do not read them as
  pure model bias without accounting for prevalence.

## Ethical considerations
No person-level scoring; human-in-the-loop required; raw narratives kept server-side;
right-sized probabilistic claims only. Subgroup metrics are suppressed below 100 rows to
avoid unstable or identifying small-group statistics.
