# Apollo — Engine (Draft 2)

Apollo predicts the **probability that a violent incident results in one or more fatalities**, from the Global Terrorism Database (GTD). It supports both fully described incidents and a location/year forecast that integrates over historically plausible incident characteristics. It does not predict incident occurrence or counts. See [`docs/intended_use.md`](docs/intended_use.md).

This is a rebuild ("draft 2") of an MSc dissertation project. The full plan is in [`STRATEGY.md`](STRATEGY.md).

## Status

| Phase | What | State |
|-------|------|-------|
| **0** | Scaffold, data registry, prediction-time contract, legal path | **done** — GTD pinned (1970–2021, hash in registry) |
| **1** | Model A (GTD-only, leakage-safe, calibrated) | **done** — ROC-AUC 0.84, Brier 0.169 on ≥2019 holdout |
| 🚦 | **Gate: Model A must beat baselines before any multi-source work** | **PASSED** (beats all baselines on AUC & Brier) |
| **2** | FastAPI inference service + Docker | **done** — serves the trained model; `/health` · `/v1/models` · `/v1/predict` verified end-to-end |
| **3** | Fairness audit + model card | **done** — subgroup audit + sensitive-field ablation; [model card](docs/model_card.md) filled |
| **4** | Model C (RAG explanation) | **done** — `/v1/explain`, grounded template + optional LLM; never alters the number |
| **5** | Honest scenario sweep | **done** — `/v1/scenario` date sweep, explicitly labelled non-forecast |
| **6** | Thin client (Android/web) | **web done** — separate repo `../apollo-web` (dependency-free dashboard over the API) |
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

**Responsible-AI audit** (Phase 3) — per-subgroup fairness (region/country/nationality) plus a
sensitive-field ablation:

```bash
pip install -e ".[fairness]"   # fairlearn
python -m training.audit       # -> reports/fairness_A.json
```

Findings are written up in the [model card](docs/model_card.md): notably, removing the
sensitive geography fields costs only ~0.02 AUC, and the model over-predicts fatality in some
regions (MENA calibration gap +0.22) — read the card before relying on any subgroup number.

## Serving — the inference API (Phase 2)

The API loads the newest `models/model_A_*.joblib` artifact **once at startup** and serves
only derived outputs — never raw GTD records (a design choice and a licence requirement).

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# or containerised (artifacts are mounted, never baked in):
#   docker build -t apollo-engine .
#   docker run -p 8000:8000 -v "$(pwd)/models:/app/models" apollo-engine
```

Endpoints:

| Route | Purpose |
|---|---|
| `GET /health` | liveness + whether a model is loaded |
| `GET /v1/models` | active version, test metrics, gate verdict, exposed feature columns |
| `POST /v1/predict` | scenario (GTD integer codes) → calibrated `P(≥1 fatality)` + uncertainty band + disclaimer |
| `POST /v1/forecast` | **year + location only** → marginal fatality prediction, distribution, and highest-risk plausible scenarios |
| `POST /v1/explain` | same scenario → the calibrated number **plus** a grounded plain-language explanation with evidence (Model C) |
| `POST /v1/scenario` | fixed scenario + year range → date-indexed probabilities, **explicitly labelled non-forecast** |

Example:

```bash
curl -s localhost:8000/v1/predict -H 'content-type: application/json' -d '{
  "iyear": 2019, "country": 95, "region": 10,
  "attacktype1": 3, "weaptype1": 6, "targtype1": 14, "suicide": 0
}'
# → {"probability": 0.634, "uncertainty_low": 0.239, "uncertainty_high": 1.0, ...}
```

The request schema uses `extra="forbid"`, so any banned outcome field (e.g. `nkill`) is
rejected with **422** at the boundary. The uncertainty band is the min–max spread across the
calibration folds — an honest indication of model disagreement, deliberately not a tight CI.
Auth (`APOLLO_API_KEY`) and rate limiting (`APOLLO_RATE_LIMIT_PER_MIN`) are opt-in via env vars.

**Location/year forecast (`/v1/forecast`).** This is the streamlined prediction
interface: the request contains only `year` and `location`. During training Apollo
stores a bounded, leakage-safe reference population plus a derived place-name index.
At inference it replaces the requested time and geography on each relevant reference
row, varies all other characteristics across that empirical population, scores every
combination, and returns the mean, quantiles, and five highest-probability plausible
points. For example: `{"year": 2050, "location": "Alaska"}`. Place names must occur
in the training data; `country:<code>` and `region:<code>` are stable fallbacks.

This is a genuine model-based prediction of severity **conditional on an incident**,
not a prediction that an incident will happen. Years after 2021 are explicit model
extrapolations. Existing artifacts must be retrained once to include the reference
population and place index.
The top-down implementation assessment and the boundary for a future occurrence
model are documented in [`docs/forecast_design.md`](docs/forecast_design.md).

**Scenario sweep (`/v1/scenario`).** The honest replacement for draft 1's fake "time series"
(flaw #5): it varies **only the date** across one fixed incident and returns the conditional
`P(≥1 fatality)` at each year (or month), holding everything else constant. Every response
carries a prominent *SCENARIO SWEEP — NOT A FORECAST* disclaimer: it does not predict whether,
where, or how often an attack occurs (that is Model B, not yet built). Sweeps are capped at 600
points; `iyear`/`imonth` are rejected as inputs because they are the swept variables.

**Model C — explanation (`/v1/explain`).** A retrieval-augmented explainer over Apollo's own
docs (intended use, model card, prediction-time contract, feature glossary), ranked by TF-IDF —
no vector DB, no external calls by default. **Hard separation:** the calibrated probability is
produced by Model A and copied verbatim; the explainer only *phrases* retrieved context and can
never change the number (guarded by a test where a lying LLM is ignored). An LLM is optional and
server-side only — set `APOLLO_LLM_PROVIDER`/`APOLLO_LLM_API_KEY`/`APOLLO_LLM_MODEL`; with none
configured, explanations use a deterministic, fully-grounded template. Keys never reach a client.

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
