# Apollo — Rebuild Strategy (Draft 2, Reconciled)

*Author's original work: MSc dissertation, University of Kent, 2024 — "Using Big Data and Deep Learning for Predicting Terrorist Incidents." This document reconciles two inputs into one plan: (a) the engine/serving rebuild strategy, and (b) the "Dataset Strategy for Draft 2." It supersedes both.*

*Last updated: 2026-07-20. Data-licensing section reflects a live terms check of GTD and ACLED on that date.*

---

## 0. TL;DR

Draft 1 proved the product idea. Draft 2 replaces the machinery, because the machinery has scientific bugs (target leakage, a "time series" that isn't one, optimistic random splitting), engineering gaps (no model persistence, train/serve encoder mismatch, ML crammed on-device), and an unbacked responsible-AI claim (fairness discussed in the thesis, implemented nowhere).

The reconciled plan is:

1. **Three models, not one vague "terrorism predictor":** Model A (conditional incident lethality), Model B (regional-month risk over time), Model C (explanation / retrieval — never alters the numbers).
2. **Prediction-time contracts** govern every feature, which is the disciplined, general fix for the leakage bug.
3. **A governed, source-aware corpus** anchored on GTD, extended by openly-licensed context sources — *not* a giant Hugging Face mixture.
4. **Train offline → version the artifact → serve via a FastAPI inference service → thin clients.** No ML on the phone.
5. **One hard gate:** *nothing multi-source gets built until Model A (GTD-only, leakage-safe, calibrated) beats its baselines.* This is the guardrail against building a beautiful data platform that never ships a model.

---

## 1. Honest reframing (do this first — it's free and it's true)

Draft 1 claimed to "predict terrorist incidents." What the core model actually does: **given the described characteristics of a hypothetical or historical incident, estimate the probability it results in one or more fatalities.** It is a *conditional lethality estimator* — retrospective, probabilistic, decision-support. Not an early-warning system for where/when an attack will occur.

Write this as a one-paragraph **intended-use statement**, and commit it to the repo, the API docs, and any client. It sets the correct evaluation target (calibrated probability), makes the timeline feature defensible (a labelled *scenario sweep*, not a forecast), and defines the model card's out-of-scope section.

---

## 2. Draft-1 flaws this rebuild must fix (verified in code)

| # | Problem | Where | Consequence |
|---|---------|-------|-------------|
| 1 | **Target leakage.** Target `death = nkill>0`, but `nkill`/`nwound` stay in the features (drop commented out) and are fed as inference inputs. | `apollo_engine.py:37`, `time_series_engine.py:48-51` | The ~86% accuracy is not trustworthy. Fatality-derived fields must never be predictors. |
| 2 | **No persistence.** Every request re-reads the CSV and retrains 4 models; `pickle` imported, unused. | `apollo_engine.py` | Unusable as an API. |
| 3 | **Train/serve encoder mismatch.** Per-column `LabelEncoder.fit_transform`, never saved; serve path feeds inconsistent codes. | `apollo_engine.py:95-97`, `time_series_engine.py:63-65` | Predictions on a different feature space than the model learned. |
| 4 | **Two unrelated models under one roof.** `generate_report` trains a separate RF on `success`, returned as `"featureImportance"`. | `apollo_controller.py:11-48` | Confusing, wasteful, mislabelled. |
| 5 | **"Time series" is not temporal.** Classifier swept over dates, all else fixed. | `time_series_engine.py` | Reframe as a scenario sweep (Model B replaces it with a real temporal model). |
| 6 | **Optimistic evaluation.** Random 80/20 + imputation stats fit on full data. | `apollo_engine.py:99-108` | Overstates generalisation. |
| 7 | **Fairness unimplemented.** | — | Central ethical claim currently unbacked. |
| 8 | **Secrets & error handling.** API key inlined; bare `except:`. | `llm_controller.py` | Security + debuggability. |

---

## 3. The three models

### Model A — Conditional incident lethality *(the v1 flagship)*
- **Question:** given an incident scenario and the information available at a declared prediction point, what is the probability of ≥1 fatality?
- **Unit:** one incident. **Label:** `total fatalities > 0`. **Optional:** severity band, civilian-fatality occurrence, count model.
- **Source:** GTD only.
- **Evaluation:** future-year holdout, calibration, Brier score, PR-AUC, per-subgroup calibration and error rates.
- **Honest ceiling:** once casualty fields are correctly removed, the signal is mostly attack × weapon × target × region. Expect the leaky 86% to become an honest ~0.70–0.80 AUC. That drop is *correctness*, not regression.

### Model B — Regional-month risk & intensity *(the v2 research effort)*
- **Question:** for a geographic unit and future month, expected probability/count/severity of violent events, using only information available before the forecast origin.
- **Unit:** country-month, admin1-month, or grid-month. **Targets:** ≥1 incident, incident count, fatal-incident count, fatalities.
- **Sources:** temporally aggregated GTD + **UCDP GED** (openly licensed). *Context:* lagged WDI, V-Dem, WorldPop.
- **Evaluation:** rolling-origin backtesting, geographic holdouts, comparison to seasonal/naive baselines.
- **Reality check:** this is a **ViEWS-class** problem (the Uppsala ViEWS program does exactly this as a multi-year funded effort). Beating seasonal-naive by a meaningful margin is genuinely hard; "we couldn't beat naive by much" is a valid scientific result. Treat Model B as research, not an engineering ticket.

### Model C — Explanation & evidence retrieval
- **Approach:** retrieval-augmented generation + structured templates over codebooks, model card, feature definitions, evaluation reports. **Do not** train a foundation model.
- **Hard separation:** Model C must never create or alter the numerical prediction. Bind its output template directly to the calibrated probability + uncertainty so narrative can't inflate a probability into operational certainty.

---

## 4. Prediction-time contracts (the disciplined leakage fix)

Every feature is assigned to a declared prediction moment. A field valid *after* an incident concludes is invalid for a pre-incident/early prediction. Maintain **separate feature sets** rather than silently mixing them.

| Feature class | Before incident | At description | After outcome | Apollo treatment |
|---|---|---|---|---|
| Date, country, region | Yes | Yes | Yes | Allowed |
| Intended target/weapon scenario | Only if supplied | Usually | Yes | Allowed with explicit contract |
| Group identity | Often unknown | Sometimes | Yes | Optional; include "unknown" honestly |
| Success of attack | No | Usually unresolved | Yes | Exclude from early prediction |
| **Fatalities / injuries / casualty totals** | No | Incomplete | Yes | **Target or post-outcome — never a predictor** |
| Property damage, hostages released, claim confirmation | Often no | Often incomplete | Yes | Exclude unless serving a post-event analytic model |

Deliverable: `docs/prediction_time_contract.md` classifying every GTD field as pre-event / scenario-known / post-event / ambiguous.

---

## 5. Data strategy

### 5.1 Dataset stack (reconciled priorities)

| Priority | Dataset | Role | Training use | Status |
|---|---|---|---|---|
| P0 | **GTD** | Canonical terrorism incident history + lethality labels | Primary event-level training set (Model A) | **Required** |
| P0 | GTD codebook + EULA | Variable definitions, provenance, legal constraints | Data contract / governance | Required |
| P1 | **UCDP GED** | Independent organized-violence history; external validation | Model B regional-month lag features; robustness | **Strongly recommended (open licence)** |
| P1 | **World Bank WDI** | Country-year population, income, urbanisation, state-capacity proxies | Lagged context features | Strongly recommended |
| P1 | **V-Dem Core** | Governance, democracy, civil-liberties context | Lagged context + fairness audits | Strongly recommended |
| P1 | **GeoNames + Natural Earth** | Stable place IDs, coordinates, boundary joins | Data engineering, not labels | Required infrastructure |
| P2 | WorldPop | Local population exposure around coordinates | Spatial features | After baseline |
| P3 | ~~ACLED~~ | Near-real-time political violence | **Excluded from training** — see §5.2 | **Blocked pending ACLED permission** |
| P3 | GDELT | Media volume/tone signals | Lagged nowcasting only | **Deprioritised — cut for v1/v2** |

Two deliberate changes from the source dataset doc, both driven by the §5.2 terms check:
- **ACLED moved from "P2 training source" to blocked-for-training.** Its EULA forbids training ML/AI on its content. Use UCDP GED for conflict context instead.
- **GDELT cut from the near-term roadmap.** Terabyte-scale, low signal-to-noise, severe look-ahead leakage risk. Don't let its P3 listing tempt you until Models A and B are validated.

### 5.2 Data licensing & deployment constraints *(verified 2026-07-20 — the section that changes decisions)*

**GTD** — [terms of use](https://www.start.umd.edu/gtd/terms-of-use/) · [FAQs](https://www.start.umd.edu/gtd-faqs)
- **Non-commercial research/analysis only.** Any commercial product/service on GTD needs a *separate* UMD agreement.
- **No redistribution/republication of raw data** — explicitly *"in any manner on any publicly-available website."*
- **Access-gated (2025):** download via request form + personal info + terms acceptance.
- **Modelling is fine; publishing raw data is not.** Apollo may serve *model outputs / derived predictions*; it must **not** expose raw GTD incident records — especially free-text narratives — through a public client or API.
- **Design consequence:** keep raw GTD (and narratives) server-side and access-controlled. If Apollo ever goes commercial or publicly hosted, the UMD agreement is a prerequisite, not an afterthought.

**ACLED** — [EULA](https://acleddata.com/eula) · [content usage](https://acleddata.com/contentusage)
- EULA **§7.1**: users *"shall not use ACLED Content… to train, test, develop, or improve any machine learning model, large language model (LLM), artificial intelligence (AI) system."* Named harms are substitute products and third-party access, but the clause is written broadly.
- Non-commercial license; commercial needs a corporate license; outputs must be transformative and not reverse-engineerable to ACLED content.
- **Decision:** do **not** train any Apollo model on ACLED without explicit written permission. If ACLED coverage is genuinely needed later, contact them describing the non-commercial research use and get it in writing first.

**Others (generally-known licences — verify at ingestion via the registry):** UCDP GED (open, academic — the safe conflict source), World Bank WDI (open API), V-Dem (academic use), GeoNames (CC-BY), Natural Earth (public domain), WorldPop (open, CC-BY-style). None independently re-verified in this session; the licence registry (§10) confirms each at download time.

### 5.3 What NOT to use as core predictive training data
General reasoning/distillation corpora; Wikipedia as labelled incident data; prompt-preference datasets; generic video/image/code datasets; any giant indiscriminate mixture. More rows never compensate for incompatible definitions, labels, time periods and observation processes. Prefer **official provider releases** over unofficial Hugging Face mirrors (the official release carries the authoritative codebook, corrections and terms).

### 5.4 Corpus architecture (when multi-source is unlocked)
Not one denormalised CSV. A small lakehouse-style corpus: immutable raw snapshots, canonical dimensions (geography spine first), task-specific feature views. Preserve every provider's original taxonomy and IDs; harmonise into a shared event *family* only for aggregation, never as a replacement for source labels. Record temporal availability (vintage) so backtests cannot see later corrections. Cross-source record linkage is a valuable *coverage-measurement* module — not training truth until manually audited.

---

## 6. Engine & serving architecture

```
                 OFFLINE (training, runs occasionally)
  GTD (pinned + hashed)  ──▶  sklearn Pipeline ──▶ eval + fairness ──▶ model artifact
   codebook/EULA/manifest     (impute→encode→          (temporal split,     (joblib) +
                               calibrated clf)          calibration, audit)  model card + metrics.json
                                                                              │ versioned in a registry
  ───────────────────────────────────────────────────────────────────────────┼─────────────────────
                 ONLINE (inference, per request)                              ▼
                                              FastAPI service — loads artifact once at startup
                                              /v1/predict · /v1/scenario · /v1/explain · /health
                                                                              │
                                    ┌─────────────────────┬───────────────────┘
                              Android thin client     web dashboard   (no ML on device)
```

Core principle: **train once, serve many.** The device never runs Python/sklearn again — exactly the recommendation from the dissertation's §3.7.

---

## 7. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Engine | Python 3.12 | ML ecosystem; keep existing language |
| API | FastAPI + Uvicorn + pydantic | Async, typed I/O, auto OpenAPI docs |
| ML | scikit-learn + LightGBM; `joblib` artifacts | Right tools for tabular data; pipeline persistence built in |
| Data | pandas (or polars if needed); pinned CSV + SHA-256; DVC optional | Reproducible versioning |
| Fairness | Fairlearn | Mature audit metrics |
| LLM (Model C) | Provider-agnostic client, **server-side keys**, structured templates | Swap models freely; keys never in the client |
| Packaging | Docker | Cloud-agnostic; runs anywhere |
| Tests | pytest — schema, model-regression, fairness-threshold, leakage-guard | CI gate |
| CI | GitHub Actions | Lint, test, build image on push |

---

## 8. API sketch

```
GET  /health                 → liveness
GET  /v1/models              → active version, metrics, model-card link
POST /v1/predict             → {scenario features} → {calibrated prob, interval, model_version, disclaimer}
POST /v1/scenario            → {features, start, end} → date-indexed probs (labelled non-forecast; Model B when ready)
POST /v1/explain             → {prediction payload} → RAG narrative (Model C, server-side)
```
Auth: API key + rate limit. Every response echoes `model_version` and a disclaimer field. No raw GTD records or narratives in any response (see §5.2).

---

## 9. Evaluation, baselines & fairness

**Split rules (non-negotiable):** no random split for the headline number; fit imputers/encoders/selection/calibration **inside** training folds only; hold out the latest GTD period as the untouched Model A test set; rolling-origin for Model B; geographic holdout stress test; deduplicate *before* splitting.

**Baselines Apollo must beat** (a complex system isn't justified otherwise): global fatality prevalence; country/attack-type prevalence with smoothing; logistic regression (one-hot); random-forest baseline; for time series — previous-month, seasonal same-month-last-year, moving average.

**Calibration is first-class:** the product surfaces a probability, so wrap the final model in `CalibratedClassifierCV` and report calibration curves + Brier alongside accuracy/F1.

**Fairness:** per-subgroup (region / country / nationality) TPR/FPR/calibration via Fairlearn; run **ablations with and without** sensitive fields (country/region/group/nationality) and publish the performance trade-off. Freeze feature selection before final evaluation; require an ablation result for every feature family.

---

## 10. Governance & responsible use

- **Licence registry** (`data/registry.yaml`): provider, version, download date, terms URL, permitted uses, attribution, redistribution limits, SHA-256, time coverage, refresh policy.
- **Row-level provenance:** every derived observation points back to source records + transformation code.
- **No person-level scoring.** Model incidents and aggregate geographic risk — never infer dangerousness of individuals, nationalities, or protected groups. The system must never recommend enforcement action against a person or community.
- **Minimum subgroup sizes** before any subgroup metric is shown.
- **Restricted raw narratives** — kept server-side, access-controlled (also a GTD licence requirement, §5.2).
- **Right-sized claims:** retrospective, probabilistic decision support — not intelligence certainty or causal proof. Human-in-the-loop framing in every response.
- **Model card + data statement** committed to the repo: intended use, out-of-scope use, provenance, per-subgroup results, limitations.

---

## 11. Reconciled roadmap (with the gate)

- **Phase 0 — Scaffold, pin, and clear the legal path.** Repo layout (`engine/`, `api/`, `training/`, `data/`, `docs/`, `tests/`); intended-use statement; **`data/registry.yaml`**; **`docs/prediction_time_contract.md`**; obtain GTD via the request form and pin the exact file + hash + codebook + EULA. Confirm the intended deployment (personal research vs. hosted vs. commercial) against GTD terms *before* building. *(No modelling yet.)*
- **Phase 1 — Model A, done right.** Leakage-safe GTD feature matrix (contracts enforced), single persisted `sklearn.Pipeline`, temporal split, calibration, honest metrics. Automated GTD ingestion report (row counts, coverage, missingness, duplicates, casualty consistency, collection-era flags).
- **🚦 GATE — Model A must beat its baselines** (§9) on the temporal holdout, with acceptable calibration, before any multi-source work begins.
- **Phase 2 — Serve it.** FastAPI loads the artifact once; `/v1/predict` + `/health`; pydantic schemas; Dockerfile; pytest incl. a leakage-guard test; CI green.
- **Phase 3 — Responsible-AI layer.** Fairness audit, sensitive-field ablations, model card, calibrated-probability output with uncertainty.
- **Phase 4 — Model C (explanation).** Server-side RAG over codebooks/model card; template bound to the calibrated output; current model, provider-agnostic.
- **Phase 5 — Honest scenario sweep.** `/v1/scenario` date-sweep, clearly labelled "non-forecast," as the interim timeline feature.
- **Phase 6 — Thin client.** Point the Android app (or a small web dashboard) at the API; remove Chaquopy/on-device ML entirely.
- **Phase 7+ (research track) — Model B.** Canonical geography dimension → country-month/admin1-month history from GTD + UCDP GED → lagged WDI/V-Dem context → rolling-origin backtesting against seasonal-naive. WorldPop after the baseline. ACLED only with written permission; GDELT only if ever justified.

Each phase is independently shippable and demoable.

---

## 12. Risks & guardrails

- **Result-honesty risk:** post-leakage accuracy drops. Frame as correctness — the old number measured the wrong thing.
- **Scope creep is the primary failure mode.** The data doc's ambition (lakehouse, linkage, rasters, multi-source) is a small-team/multi-quarter effort. The Phase-1 gate exists to stop a beautiful-platform-no-model outcome. Resist Model B, WorldPop, and GDELT until the gate is passed.
- **Legal exposure:** GTD forbids republishing raw data publicly and restricts commercial use; ACLED forbids ML training on its content. Keep raw data server-side; keep ACLED out of training; revisit terms at every production refresh (they change).
- **Data staleness:** GTD effectively ends ~2021; document the training window honestly and don't claim currency you don't have.
- **Model B expectation-setting:** it may not beat seasonal-naive by much. That's science, not failure — report it either way.
