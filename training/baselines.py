"""Baselines Apollo Model A must beat (STRATEGY.md §9).

A complex model is not justified unless it materially improves calibrated
performance over these. All baselines output a probability so they can be scored
on the same footing (Brier, ROC-AUC, PR-AUC) as the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PrevalenceBaseline:
    """Predicts the global training positive rate for every row."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PrevalenceBaseline:
        self.rate_ = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.rate_)


class GroupPrevalenceBaseline:
    """Predicts smoothed P(y=1 | group) for a single categorical column.

    Laplace-style smoothing toward the global rate so rare groups don't overfit.
    """

    def __init__(self, group_col: str, alpha: float = 10.0):
        self.group_col = group_col
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GroupPrevalenceBaseline:
        self.global_ = float(y.mean())
        g = y.groupby(X[self.group_col])
        counts, sums = g.count(), g.sum()
        self.rates_ = ((sums + self.alpha * self.global_) / (counts + self.alpha)).to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.group_col].map(self.rates_).fillna(self.global_).to_numpy(dtype=float)


def default_baselines(feature_columns: list[str]) -> dict[str, object]:
    """Standard baseline set; group baseline keyed on attack type when available."""
    out: dict[str, object] = {"prevalence": PrevalenceBaseline()}
    for col in ("attacktype1", "country", "region"):
        if col in feature_columns:
            out[f"group_prevalence[{col}]"] = GroupPrevalenceBaseline(col)
            break
    return out
