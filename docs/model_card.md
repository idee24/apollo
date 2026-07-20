# Model Card — Apollo Model A (SKELETON)

> Fill during Phase 3. Do not publish any model without a completed card.
> Template follows Mitchell et al., "Model Cards for Model Reporting."

## Model details
- **Name / version:** _TBD_
- **Date:** _TBD_
- **Type:** Binary classifier — probability of ≥1 fatality given an incident scenario.
- **Algorithm:** _TBD (RF baseline / LightGBM challenger), wrapped in probability calibration._
- **Training data:** GTD `<version>`, pinned hash `<sha256>`. See `data/registry.yaml`.
- **Features:** per `docs/prediction_time_contract.md` (PRE_EVENT + SCENARIO only). Outcome fields excluded by `engine/leakage.py`.

## Intended use
See `docs/intended_use.md`. Retrospective, probabilistic decision-support. **Not** early warning, **not** causal, **not** person-scoring.

## Out-of-scope / prohibited use
- Predicting where/when an attack will occur.
- Any action against an individual, nationality, or protected group.
- Commercial or public redistribution of raw GTD data.

## Evaluation
- **Split:** temporal holdout (latest GTD period untouched). No random split for headline numbers.
- **Metrics:** _TBD_ — accuracy, F1, PR-AUC, **Brier score + calibration curve**.
- **Baselines beaten:** prevalence, country/attack-type prevalence, logistic regression, RF. _(fill results)_

## Fairness / subgroup analysis
- **Subgroups:** region / country / (nationality where meaningful). Minimum group size enforced.
- **Metrics per subgroup:** _TBD_ — calibration, TPR, FPR.
- **Sensitive-field ablation:** performance with vs. without country/region/group/nationality. _(fill trade-off)_

## Limitations
- Data ends ~2021; reporting intensity varies by era/region.
- Modest signal after correct removal of casualty fields; honest AUC materially below any leaky figure.

## Ethical considerations
No person-level scoring; human-in-the-loop required; raw narratives kept server-side; right-sized probabilistic claims only.
