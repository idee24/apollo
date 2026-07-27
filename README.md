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

## Data — you must obtain GTD yourself

GTD is **access-gated and non-redistributable** (non-commercial research only; raw data may not be republished on any public site). Apollo does not and cannot ship it. Request it via the GTD download form, then place the file under `data/raw/` and record its hash in `data/registry.yaml`. See [`data/raw/README.md`](data/raw/README.md) and `STRATEGY.md` §5.2.

## Non-negotiables (why draft 2 exists)

- **No fatality/casualty/outcome field is ever a predictor** of the fatality target. Enforced in code by [`engine/leakage.py`](engine/leakage.py) and guarded by tests.
- **Temporal evaluation**, not random splits. Calibration is first-class.
- **No person-level scoring.** Aggregate/incident risk only. Never recommends action against a person or group.
- **Raw data and incident narratives stay server-side.** Only model outputs are served.
