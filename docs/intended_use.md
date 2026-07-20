# Apollo — Intended Use Statement

## What Apollo is

Apollo estimates, for a **described violent-incident scenario**, the probability that the incident results in **one or more fatalities**, based on historical patterns in the Global Terrorism Database (GTD, 1970–2021). It is a **conditional lethality estimator**: given the characteristics of a hypothetical or historical incident and the information available at a declared prediction point, it returns a *calibrated probability with an uncertainty range*, plus a plain-language explanation.

Its purpose is **retrospective, probabilistic decision-support** for research and for resource-planning conversations — for example, understanding how estimated lethality varies across attack types, weapons, targets and regions.

## What Apollo is NOT

- **Not an early-warning system.** It does not predict *where* or *when* an attack will occur. (A separate research track — "Model B" — attempts genuine regional-month risk forecasting and is held to a much higher evidentiary bar; it is not this model.)
- **Not intelligence certainty.** Outputs are probabilities, not facts. A high probability is not a prediction that an event will happen; a low probability is not a guarantee of safety.
- **Not causal.** Apollo describes statistical association in historical data. It does not establish that any factor *causes* fatalities.
- **Not a person-scoring tool.** Apollo models incidents and aggregate geographic risk. It must never be used to infer the dangerousness of an individual, a nationality, or any protected group, and must never recommend enforcement or other action against a person or community.

## How outputs must be presented

Every prediction is accompanied by: the model version, a calibrated probability **with an uncertainty interval**, and an explicit disclaimer that the output is retrospective, probabilistic decision-support requiring human judgement and corroborating evidence. Raw GTD incident records and free-text narratives are **not** exposed to end users (both a design choice and a GTD licence requirement).

## Intended users

Researchers, students, and analysts who understand probabilistic model output and its limits. Any operational use requires a human decision-maker who treats Apollo as one input among many, never as an authority.

## Known limitations

- Trained on historical data (GTD effectively ends ~2021); patterns may not reflect current dynamics.
- Reporting intensity and coding practices in GTD vary across decades and regions; the data is a record of *reported* incidents, not ground truth.
- After the necessary removal of casualty/outcome fields (to prevent target leakage), the available predictive signal is modest. Honest performance is materially lower than any figure produced by a leaky model.
