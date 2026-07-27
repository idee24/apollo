# Apollo — Engine (Draft 2)

Apollo estimates the **probability that a described violent incident results in one or more fatalities**, from the Global Terrorism Database (GTD). It is a *conditional lethality estimator* — retrospective, probabilistic, decision-support — **not** an early-warning system. See [`docs/intended_use.md`](docs/intended_use.md).

This is a rebuild ("draft 2") of an MSc dissertation project. The full plan is in [`STRATEGY.md`](STRATEGY.md).

## Status

| Phase | What | State |
|-------|------|-------|
| **0** | Scaffold, data registry, prediction-time contract, legal path | **done** — GTD pinned (1970–2021, hash in registry) |
| **1** | Model A (GTD-only, leakage-safe, calibrated) | **done** — ROC-AUC 0.84, Brier 0.169 on ≥2019 holdout |
| 🚦 | **Gate: Model A must beat baselines before any multi-source work** | **PASSED** (beats all baselines on AUC & Brier) |
| 2 | FastAPI inference service + Docker | code scaffolded — wire artifact loading next |
| 3 | Fairness audit + model card | not started |
| 4 | Model C (RAG explanation) | not started |
| 5 | Honest scenario sweep | not started |
| 6 | Thin client (Android/web) | not started |
| 7+ | Model B (regional-month risk — research track) | not started |

## Layout

```
apollo/
├── STRATEGY.md          # the reconciled plan (read this first)
├── README.md
├── pyproject.toml
├── data/
│   ├── registry.yaml    # every dataset: version, terms, hash, training-use
│   └── raw/             # GTD & other raw files — GITIGNORED (licence: never commit/redistribute)
├── docs/
│   ├── intended_use.md          # what Apollo claims / does not claim
│   ├── prediction_time_contract.md  # every field: pre-event / scenario / post-outcome (anti-leakage)
│   └── model_card.md            # skeleton, filled during Phase 3
├── engine/              # inference-time code + shared contracts (leakage guard lives here)
├── training/            # offline training pipeline (Phase 1)
├── api/                 # FastAPI service (Phase 2)
├── tests/
└── python/              # DRAFT-1 reference code — do not build on; kept for provenance
```

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Git Bash:            source .venv/Scripts/activate
pip install -e ".[dev]"
pytest
```

## Training — Model A

With GTD pinned under `data/raw/gtd/` (see below), train, calibrate, evaluate against
the required baselines, and persist the artifact:

```bash
python -m training.train
```

This writes a calibrated model to `models/` (gitignored) and a metrics report to
[`reports/model_A_metrics.json`](reports/model_A_metrics.json). Training **refuses to run** unless the data
file's SHA-256 matches the registry.

**Current result** (untouched ≥2019 temporal holdout, 20,253 incidents): ROC-AUC **0.840**,
PR-AUC 0.856, Brier **0.169** — beats every required baseline on both AUC and Brier, so the
Phase-1 gate is **passed**. This honest ~0.84 replaces draft-1's leaky 86% accuracy.

## Data — you must obtain GTD yourself

GTD is **access-gated and non-redistributable** (non-commercial research only; raw data may not be republished on any public site). Apollo does not and cannot ship it. Request it via the GTD download form, then place the files under `data/raw/gtd/` (gitignored) and pin them in [`data/registry.yaml`](data/registry.yaml). See [`data/raw/README.md`](data/raw/README.md) and `STRATEGY.md` §5.2.

The pinned dataset for the current model is a single **UTF-8 CSV covering 1970–2021**
(214,666 rows), derived once from the official `.xlsx` releases — the May-2022 main file
(1970–2020) plus the 2021 H1 supplement — concatenated (identical 135-column schema). The
one-off conversion needs the ingest extra (`pip install -e ".[ingest]"`, for `openpyxl`);
the resulting CSV's hash is recorded in the registry so the pinned data can't silently change.

## Non-negotiables (why draft 2 exists)

- **No fatality/casualty/outcome field is ever a predictor** of the fatality target. Enforced in code by [`engine/leakage.py`](engine/leakage.py) and guarded by tests.
- **Temporal evaluation**, not random splits. Calibration is first-class.
- **No person-level scoring.** Aggregate/incident risk only. Never recommends action against a person or group.
- **Raw data and incident narratives stay server-side.** Only model outputs are served.
