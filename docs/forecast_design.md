# Location/year prediction design

## Top-down assessment

Apollo has two Python generations. `python/` is the retained draft-1 reference and
must not be imported by production code. It mixes ingestion, preprocessing,
training, prediction, reporting, and an LLM call in one controller; uses a random
split; includes post-outcome fields; and creates a date series by holding a single
incident fixed. Those properties make its apparent forecast unreliable.

The maintained implementation separates responsibilities:

1. `training/` verifies the licensed dataset, constructs leakage-safe features,
   performs a temporal split, calibrates and evaluates the model, and persists one
   inference artifact.
2. `engine/` owns shared feature and leakage contracts.
3. `api/` validates requests and exposes prediction, explanation, scenario, and
   forecast services. The optional LLM is confined to explanation and cannot
   alter a probability.
4. `tests/` trains synthetic artifacts so boundaries can be checked without
   redistributing GTD.

## Forecast semantics

The forecast accepts exactly a year and location. Training saves a reproducible
geographically stratified sample of up to 10,000 leakage-safe feature rows and a
derived country/province place-name catalog. The catalog is built from the complete
training partition so uncommon locations are not randomly dropped; ambiguous names
are rejected rather than silently assigned to the wrong geography.
Inference selects the relevant geographic population, overwrites only time and
location, and scores the remaining empirical feature combinations. This is
Monte-Carlo-style empirical marginalisation rather than single-value imputation.
The response reports the mean prediction, scenario-spread quantiles, sample size,
and up to five distinct highest-probability evaluated combinations for inspection.
It also reports `population_basis` — the geography whose historical characteristic
mix was marginalised over (`country:<code>` / `region:<code>`), or `global` when a
requested place had too few local reference rows (< 25) and the global empirical
distribution was used instead. That flag matters because the reported expectation
is an average over *that* population's incident mix, so the two cases are not
directly comparable and the caller should be able to tell them apart.

This architecture removes the need for a caller to invent attack, weapon, target,
nationality, or suicide values. It does **not** remove outcome-leakage protection:
an outcome field would make evaluation look stronger while making a future
prediction impossible.

## Artifact data boundary

Persisting `forecast_reference` changes the artifact's data-sensitivity profile.
Before this feature the artifact held only fitted model parameters; it now also
embeds a bounded (≤ 10,000-row) verbatim sample of leakage-safe GTD **feature**
rows — including `latitude`/`longitude` and geography/attack codes. No outcome,
casualty, narrative, or source fields are included (the leakage guard runs over
the scored columns), and the API never returns these rows: `/v1/forecast` exposes
only derived aggregates and generic scenario sentences with `iyear`/`country`
overwritten, so no individual training incident is reconstructable from a response.

Nonetheless the artifact now contains a sample of licensed GTD records, so it
inherits the dataset's non-redistribution constraint (see the project licence in
`pyproject.toml` and the "never raw GTD records" note in `api/main.py`). The
practical invariant: **the model artifact must stay server-side and must never be
shipped to any client, including the web thin client.** Existing artifacts must be
retrained once to gain forecast support.

## Current boundary and next model

The current target remains fatality conditional on an incident. Therefore the
location/year result is a severity forecast, not an incident-occurrence or event-
count forecast. Years after the GTD training endpoint are extrapolations and are
labelled as such. A forecast of total regional risk needs a separately evaluated
panel model with explicit zero-incident region/month rows, exposure denominators,
rolling-origin validation, and an occurrence/count target. Multiplying that model's
event-rate distribution by this model's conditional severity distribution would
support an unconditional expected-fatality forecast without changing this target.
