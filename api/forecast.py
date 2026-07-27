"""Location/year forecast by marginalising unknown incident characteristics.

This is deliberately different from filling missing columns with model-wide
medians.  The engine evaluates an empirical population of historically plausible
feature combinations and aggregates their calibrated probabilities.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from engine.codes import describe_scenario
from engine.leakage import assert_leakage_safe
from engine.locations import normalize_location

MAX_EVALUATIONS = 5000
DISCLAIMER = (
    "Model-based forecast of fatality conditional on a violent incident occurring; it does "
    "not forecast whether or how many incidents will occur. Future years are extrapolations "
    "beyond the 1970-2021 training period. Do not use it to act against a person or group."
)


def _resolve_location(bundle: dict, location: str) -> dict:
    catalog = bundle.get("location_catalog", {})
    key = normalize_location(location)
    if key in catalog:
        candidates = catalog[key]
        # Compatibility with artifacts produced by the first forecast prototype.
        if isinstance(candidates, dict):
            return candidates
        if len(candidates) == 1:
            return candidates[0]
        labels = ", ".join(
            f"{item['label']} (country:{item.get('country', '?')})" for item in candidates
        )
        raise ValueError(
            f"Ambiguous location {location!r}: {labels}. Use 'country:<GTD code>'."
        )
    # Stable machine-readable fallbacks are useful when a licensed artifact was
    # trained without text location columns.
    match = re.fullmatch(r"(country|region)\s+(\d+)", key)
    if match:
        return {match.group(1): int(match.group(2)), "label": location}
    raise ValueError(
        f"Unknown location {location!r}. Use a place present in the training data, "
        "or 'country:<GTD code>' / 'region:<GTD code>'."
    )


def forecast(bundle: dict, year: int, location: str) -> dict:
    reference = bundle.get("forecast_reference")
    if reference is None or len(reference) == 0:
        raise ValueError("This artifact predates forecast support; retrain Model A.")
    reference = pd.DataFrame(reference).copy()
    resolved = _resolve_location(bundle, location)
    cols = list(bundle["numeric"]) + list(bundle["categorical"])
    assert_leakage_safe(cols)

    # Prefer historical rows from the same geography. Fall back to the global
    # empirical distribution if a textual alias has coordinates but few examples.
    # `population_basis` records which one we used so the caller can see how the
    # reported expectation is grounded (a local country/region mix vs. global).
    local = reference
    population_basis = "global"
    for field in ("country", "region"):
        if field in resolved and field in local:
            candidate = local[local[field] == resolved[field]]
            if len(candidate) >= 25:
                local = candidate
                population_basis = f"{field}:{resolved[field]}"
                break
    if len(local) > MAX_EVALUATIONS:
        local = local.sample(MAX_EVALUATIONS, random_state=42)

    X = local.reindex(columns=cols)
    if "iyear" in X:
        X["iyear"] = year
    for field in ("country", "region", "latitude", "longitude"):
        if field in X and resolved.get(field) is not None:
            X[field] = resolved[field]
    model = bundle["model"]
    probabilities = model.predict_proba(X)[:, 1]

    # Highest-probability plausible scenarios, distinct on their reported codes so
    # the list is five *different* scenarios for inspection rather than five
    # near-identical rows that happen to top the ranking.
    positive_points = []
    seen: set[tuple] = set()
    for idx in np.argsort(probabilities)[::-1]:
        idx = int(idx)
        features = {k: X.iloc[idx][k] for k in cols if pd.notna(X.iloc[idx][k])}
        reported = {
            k: int(features[k]) if float(features[k]).is_integer() else float(features[k])
            for k in ("attacktype1", "weaptype1", "targtype1", "suicide")
            if k in features
        }
        signature = tuple(sorted(reported.items()))
        if signature in seen:
            continue
        seen.add(signature)
        positive_points.append({
            "probability": round(float(probabilities[idx]), 4),
            "scenario": describe_scenario(features),
            "features": reported,
        })
        if len(positive_points) == 5:
            break

    q05, q25, q50, q75, q95 = np.quantile(probabilities, [.05, .25, .5, .75, .95])
    return {
        "year": year,
        "location": resolved.get("label", location),
        "probability": round(float(np.mean(probabilities)), 4),
        "distribution": {
            "p05": round(float(q05), 4), "p25": round(float(q25), 4),
            "median": round(float(q50), 4), "p75": round(float(q75), 4),
            "p95": round(float(q95), 4),
        },
        "distribution_description": (
            "Percentiles across plausible incident scenarios; not a confidence interval."
        ),
        "population_basis": population_basis,
        "scenarios_evaluated": len(X),
        "positive_points": positive_points,
        "target": (
            "Expected P(>=1 fatality), conditional on an incident, across plausible scenarios."
        ),
        "method": "Empirical marginalisation over training-era incident characteristics.",
        "disclaimer": DISCLAIMER,
    }
