"""Fairness-module unit tests (synthetic; no real GTD needed)."""

import numpy as np

from training.fairness import reliability_curve, subgroup_report


def test_reliability_curve_bins_partition_all_rows():
    rng = np.random.default_rng(0)
    proba = rng.random(500)
    y = rng.binomial(1, proba)
    curve = reliability_curve(y, proba, n_bins=10)
    assert len(curve) == 10
    assert sum(b["n"] for b in curve) == 500
    # A perfectly-predicting proba should have frac_positive tracking mean_pred loosely.
    for b in curve:
        if b["n"]:
            assert 0.0 <= b["mean_pred"] <= 1.0


def test_subgroup_report_suppresses_small_groups():
    # Group A: 300 rows (kept). Group B: 10 rows (suppressed at min_size=100).
    y = np.array([1, 0] * 150 + [1, 0] * 5)
    proba = np.array([0.7, 0.3] * 150 + [0.6, 0.4] * 5)
    groups = np.array(["A"] * 300 + ["B"] * 10)
    rep = subgroup_report(y, proba, groups, min_size=100)
    assert "A" in rep["groups"]
    assert "B" not in rep["groups"]  # withheld, never shown
    assert rep["suppressed"]["n_groups"] == 1
    assert rep["suppressed"]["n_rows"] == 10
    assert rep["n_groups_reported"] == 1


def test_subgroup_metrics_and_disparities():
    rng = np.random.default_rng(1)
    n = 600
    # Two well-populated groups with different base rates -> real disparity.
    ga = {"y": rng.binomial(1, 0.7, n), "p": rng.uniform(0.5, 0.9, n)}
    gb = {"y": rng.binomial(1, 0.2, n), "p": rng.uniform(0.1, 0.5, n)}
    y = np.concatenate([ga["y"], gb["y"]])
    p = np.concatenate([ga["p"], gb["p"]])
    g = np.array(["high"] * n + ["low"] * n)
    rep = subgroup_report(y, p, g, min_size=100)
    assert set(rep["groups"]) == {"high", "low"}
    for m in rep["groups"].values():
        for k in ("n", "positive_rate", "tpr", "fpr", "brier", "calibration_gap"):
            assert k in m
    d = rep["disparities"]
    assert d["tpr"]["range"] is not None
    assert d["tpr"]["range"] >= 0
    assert "worst_calibration_gap" in d


def test_nan_groups_become_unknown_not_dropped():
    y = np.array([1, 0] * 100)
    p = np.array([0.6, 0.4] * 100)
    g = np.array([np.nan] * 200, dtype=object)
    rep = subgroup_report(y, p, g, min_size=50)
    assert "unknown" in rep["groups"]
