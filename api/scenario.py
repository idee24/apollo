"""Scenario sweep (Phase 5) — the honest replacement for draft 1's fake time series.

Draft 1 swept a classifier over dates with everything else fixed and presented it
as a forecast (STRATEGY.md flaw #5). This does the same mechanical sweep but frames
it correctly: it shows how Model A's *conditional* probability of >=1 fatality for
the SAME described incident changes as only the date varies. It is NOT a forecast
of whether, where, or how often an attack occurs — that is Model B (not yet built).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.leakage import assert_leakage_safe

MAX_POINTS = 600  # cap the sweep so a request can't fan out unboundedly

DISCLAIMER = (
    "SCENARIO SWEEP - NOT A FORECAST. This shows how Model A's conditional probability "
    "of >=1 fatality for the SAME described incident changes as only the date is varied, "
    "with every other characteristic held fixed. It does not predict whether, where, or "
    "how often an attack will occur on any date, and the trend is not a time series. A "
    "genuine temporal forecast is a separate research model (Model B), not yet available. "
    "Retrospective, probabilistic decision-support only; requires human judgement."
)


def build_points(start_year: int, end_year: int, by: str, month: int | None) -> list[dict]:
    """Date overrides for the sweep. by='year' -> one point per year (imonth fixed
    to `month` or left missing); by='month' -> one point per (year, month)."""
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")
    points: list[dict] = []
    for y in range(start_year, end_year + 1):
        if by == "month":
            for m in range(1, 13):
                points.append({"date": f"{y}-{m:02d}", "iyear": y, "imonth": m})
        else:
            p = {"date": str(y), "iyear": y}
            if month is not None:
                p["imonth"] = month
            points.append(p)
    if not points:
        raise ValueError("empty sweep range")
    if len(points) > MAX_POINTS:
        raise ValueError(f"sweep too large ({len(points)} points; max {MAX_POINTS})")
    return points


def sweep(bundle: dict, base_features: dict, points: list[dict]) -> list[dict]:
    """Evaluate the calibrated model at each swept date, holding all else fixed."""
    cols = list(bundle["numeric"]) + list(bundle["categorical"])
    rows = []
    for pt in points:
        row = {c: pt.get(c, base_features.get(c, np.nan)) for c in cols}
        rows.append(row)
    X = pd.DataFrame(rows, columns=cols)
    assert_leakage_safe(X.columns)  # date-only sweep, but guard the boundary anyway

    model = bundle["model"]
    proba = model.predict_proba(X)[:, 1]

    folds = getattr(model, "calibrated_classifiers_", None)
    if folds:
        fold_probs = np.stack([cc.predict_proba(X)[:, 1] for cc in folds])  # (F, N)
        lo = fold_probs.min(axis=0)
        hi = fold_probs.max(axis=0)
    else:
        lo = hi = proba

    return [
        {
            "date": pt["date"],
            "probability": round(float(proba[i]), 4),
            "uncertainty_low": round(float(lo[i]), 4),
            "uncertainty_high": round(float(hi[i]), 4),
        }
        for i, pt in enumerate(points)
    ]
