"""Per-subgroup fairness metrics and calibration diagnostics (Phase 3).

STRATEGY.md §9/§10: report per-subgroup TPR/FPR/selection-rate (via Fairlearn)
and per-subgroup calibration, but never below a minimum group size. This module
computes the numbers only; it takes no view on what an acceptable disparity is —
that judgement belongs in the model card and to a human reviewer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import roc_auc_score

from training.config import MIN_GROUP_SIZE


def _r(x, nd: int = 4):
    """Round to a JSON-friendly float, or None for NaN/undefined."""
    if x is None:
        return None
    xf = float(x)
    return None if np.isnan(xf) else round(xf, nd)


def reliability_curve(y_true, proba, *, n_bins: int = 10) -> list[dict]:
    """Calibration curve as fixed-width probability bins (STRATEGY.md §9)."""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        out.append({
            "bin": f"[{edges[b]:.1f},{edges[b + 1]:.1f})",
            "n": n,
            "mean_pred": _r(proba[m].mean()) if n else None,
            "frac_positive": _r(y_true[m].mean()) if n else None,
        })
    return out


def subgroup_report(
    y_true, proba, groups, *, min_size: int = MIN_GROUP_SIZE, threshold: float = 0.5
) -> dict:
    """Per-group metrics with small-group suppression.

    Returns ``{min_size, groups, suppressed, disparities}``. ``groups`` maps each
    kept group label to n / positive_rate / selection_rate / tpr / fpr / brier /
    calibration_gap / roc_auc. Groups below ``min_size`` are withheld (only their
    count and total rows are reported) — never shown, per STRATEGY.md §10.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    g = pd.Series(np.asarray(groups)).fillna("unknown").astype(str).reset_index(drop=True)

    # Fairlearn owns the classification-rate metrics.
    mf = MetricFrame(
        metrics={"tpr": true_positive_rate, "fpr": false_positive_rate,
                 "selection_rate": selection_rate},
        y_true=y_true, y_pred=pred, sensitive_features=g,
    )
    by = mf.by_group

    df = pd.DataFrame({"y": y_true, "p": proba, "g": g.to_numpy()})
    kept: dict[str, dict] = {}
    suppressed_groups = 0
    suppressed_rows = 0
    for grp, sub in df.groupby("g", sort=True):
        n = len(sub)
        if n < min_size:
            suppressed_groups += 1
            suppressed_rows += n
            continue
        kept[str(grp)] = {
            "n": int(n),
            "positive_rate": _r(sub.y.mean()),
            "selection_rate": _r(by.loc[grp, "selection_rate"]),
            "tpr": _r(by.loc[grp, "tpr"]),
            "fpr": _r(by.loc[grp, "fpr"]),
            "brier": _r(np.mean((sub.p - sub.y) ** 2)),
            "calibration_gap": _r(sub.p.mean() - sub.y.mean()),  # + = over-predicts
            "roc_auc": _r(roc_auc_score(sub.y, sub.p)) if sub.y.nunique() > 1 else None,
        }

    return {
        "min_size": min_size,
        "n_groups_reported": len(kept),
        "suppressed": {"n_groups": suppressed_groups, "n_rows": int(suppressed_rows)},
        "disparities": _disparities(kept),
        "groups": kept,
    }


def _disparities(kept: dict[str, dict]) -> dict:
    """Spread across reported groups: max-min for rates, worst absolute
    calibration gap. Empty when fewer than two groups clear ``min_size``."""
    if len(kept) < 2:
        return {}

    def spread(metric: str) -> dict | None:
        vals = {g: r[metric] for g, r in kept.items() if r.get(metric) is not None}
        if len(vals) < 2:
            return None
        hi_g = max(vals, key=vals.get)
        lo_g = min(vals, key=vals.get)
        return {"max": {"group": hi_g, "value": vals[hi_g]},
                "min": {"group": lo_g, "value": vals[lo_g]},
                "range": _r(vals[hi_g] - vals[lo_g])}

    worst_cal = max(kept.items(), key=lambda kv: abs(kv[1].get("calibration_gap") or 0.0))
    return {
        "tpr": spread("tpr"),
        "fpr": spread("fpr"),
        "roc_auc": spread("roc_auc"),
        "worst_calibration_gap": {"group": worst_cal[0],
                                  "calibration_gap": worst_cal[1].get("calibration_gap")},
    }
