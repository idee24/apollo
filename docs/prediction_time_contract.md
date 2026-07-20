# Prediction-Time Contract (GTD)

**Purpose:** every feature is assigned to a *declared prediction moment*. A field that only becomes known *after* an incident concludes must never be used to predict that incident's outcome. This document is the anti-leakage specification; the canonical machine-readable list lives in [`engine/leakage.py`](../engine/leakage.py) and is enforced by tests.

**The target (Model A):** `death = (total fatalities > 0)`. Therefore **all fatality-, casualty-, and outcome-derived fields are the label's basis or its siblings and can never be predictors.** This is the exact bug that inflated draft 1.

> ⚠️ **Verification required.** The classifications below are a first pass built from the draft-1 code and the GTD codebook (Methodology, Aug 2021). Before Phase 1 modelling is finalised, reconcile every GTD column against the official codebook and move any `AMBIGUOUS` field to a definite class. Do not add a field to the model until it has a definite class.

---

## Class definitions

| Class | Meaning | Usable as predictor for Model A? |
|-------|---------|----------------------------------|
| `PRE_EVENT` | Known independent of the incident (date, place, context) | Yes |
| `SCENARIO` | Attack characteristics supplied *as the scenario* to be scored | Yes (they are the user's input) |
| `AMBIGUOUS` | Sometimes known at description time, often only confirmed later | **No** until resolved to a definite class per-use-case |
| `POST_OUTCOME` | Known only after the incident resolves | **Never** |

---

## PRE_EVENT — allowed
- `eventid`, `iyear`, `imonth`, `iday` *(date; `iday`/`imonth` can be 0 = unknown — treat as missing)*
- `country`, `country_txt`, `region`, `region_txt`, `provstate`, `city`
- `latitude`, `longitude`, `specificity`, `vicinity`
- *Context joins added later:* lagged WDI / V-Dem country-year values, population/exposure — all **lagged** so no value from on/after the incident date leaks in.

## SCENARIO — allowed (these ARE the model's input at scoring time)
- `attacktype1/2/3` (+ `_txt`)
- `targtype1/2/3`, `targsubtype1/2/3` (+ `_txt`), `corp1`, `target1`
- `natlty1/2/3` (+ `_txt`) — nationality of target
- `weaptype1..4`, `weapsubtype1..4` (+ `_txt`)
- `suicide` — attack modality
- `multiple` — part of a coordinated attack *(only if supplied as scenario; can be coded later — treat with care)*

## AMBIGUOUS — excluded until resolved per use-case
- `gname`, `gsubname`, `guncertain1`, `individual` — perpetrator identity (often confirmed post-event; include only with an explicit contract, and always allow an honest "unknown")
- `nperps`, `nperpcap` — perpetrator counts (frequently post-event)
- `claimed`, `claimmode` (+ `_txt`) — claim of responsibility (post-event)
- `weapdetail`, `summary` — free text authored post-hoc *(also narrative → keep server-side)*
- `doubtterr`, `crit1`, `crit2`, `crit3` — coding/inclusion artifacts, not real-world predictors
- `INT_LOG`, `INT_IDEO`, `INT_MISC`, `INT_ANY` — international-attribution codes assigned in post-event analysis *(draft 1 wrongly used these as inputs)*

## POST_OUTCOME — NEVER predictors (banned)
- **`nkill`, `nkillus`, `nkillter`** — fatality counts = the label basis
- **`nwound`, `nwoundus`, `nwoundte`** — casualty counts
- `success` — attack outcome
- `property`, `propextent` (+ `_txt`), `propvalue`, `propcomment` — damage outcome
- `ishostkid`, `nhostkid`, `nhostkidus`, `nhours`, `ndays`, `divert`, `kidhijcountry`, `ransom`, `ransomamt`, `ransomamtus`, `ransompaid`, `ransompaidus`, `ransomnote`, `hostkidoutcome` (+ `_txt`), `nreleased` — hostage/kidnap outcomes
- `addnotes`, `scite1/2/3`, `dbsource`, `related` — source/coding metadata

---

## Rules

1. **The banned (`POST_OUTCOME`) set is defined once** in `engine/leakage.py`. The training pipeline calls the leakage guard on its feature matrix and **fails loudly** if any banned field survives.
2. **`AMBIGUOUS` fields are excluded by default.** Promoting one to a predictor requires a written note here stating for which prediction moment it is legitimately known.
3. **Context features must be lagged.** Any country-year/month join uses values dated strictly *before* the incident/forecast date.
4. When in doubt, a field is `POST_OUTCOME`. Leakage is a silent failure; exclusion is a visible, safe one.
