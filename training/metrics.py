"""Probability-first evaluation metrics (STRATEGY.md §9)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def evaluate_proba(y_true, proba, *, threshold: float = 0.5) -> dict:
    """Accuracy/F1 plus the metrics that actually matter for a probability output:
    ROC-AUC, PR-AUC (average precision), and Brier score (calibration)."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 4),
        "accuracy": round(float((pred == y_true).mean()), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
    }
    # AUC metrics require both classes present.
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_true, proba)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_true, proba)), 4)
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out
