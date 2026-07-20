"""Turn a validated scenario into a calibrated prediction.

The uncertainty band is the min–max spread of the per-fold calibrated
classifiers inside CalibratedClassifierCV — an honest indication of model
disagreement, not a formal confidence interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.leakage import assert_leakage_safe

TARGET_DESCRIPTION = (
    "Calibrated probability of >=1 fatality in the described incident scenario."
)
DISCLAIMER = (
    "Retrospective, probabilistic decision-support only. Not a forecast of whether "
    "an attack will occur, not causal, and never a basis for action against any "
    "person or group. Requires human judgement and corroborating evidence."
)


def _build_row(bundle: dict, features: dict) -> pd.DataFrame:
    cols = list(bundle["numeric"]) + list(bundle["categorical"])
    row = {c: features.get(c, np.nan) for c in cols}
    X = pd.DataFrame([row], columns=cols)
    assert_leakage_safe(X.columns)  # boundary tripwire
    return X


def _interval(model, X: pd.DataFrame, mean: float) -> tuple[float, float]:
    folds = getattr(model, "calibrated_classifiers_", None)
    if not folds:
        return mean, mean
    ps = [float(cc.predict_proba(X)[0, 1]) for cc in folds]
    return min(ps), max(ps)


def predict(bundle: dict, features: dict) -> dict:
    X = _build_row(bundle, features)
    model = bundle["model"]
    proba = float(model.predict_proba(X)[0, 1])
    lo, hi = _interval(model, X, proba)
    return {
        "probability": round(proba, 4),
        "uncertainty_low": round(lo, 4),
        "uncertainty_high": round(hi, 4),
        "target": TARGET_DESCRIPTION,
        "disclaimer": DISCLAIMER,
    }
