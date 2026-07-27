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
sample of up to 10,000 leakage-safe feature rows and a derived place-name catalog.
Inference selects the relevant geographic population, overwrites only time and
location, and scores the remaining empirical feature combinations. This is
Monte-Carlo-style empirical marginalisation rather than single-value imputation.
The response reports the mean prediction, distribution quantiles, sample size,
and the five highest-probability evaluated combinations for inspection.

This architecture removes the need for a caller to invent attack, weapon, target,
nationality, or suicide values. It does **not** remove outcome-leakage protection:
an outcome field would make evaluation look stronger while making a future
prediction impossible.

## Current boundary and next model

The current target remains fatality conditional on an incident. Therefore the
location/year result is a severity forecast, not an incident-occurrence or event-
count forecast. Years after the GTD training endpoint are extrapolations and are
labelled as such. A forecast of total regional risk needs a separately evaluated
panel model with explicit zero-incident region/month rows, exposure denominators,
rolling-origin validation, and an occurrence/count target. Multiplying that model's
event-rate distribution by this model's conditional severity distribution would
support an unconditional expected-fatality forecast without changing this target.
