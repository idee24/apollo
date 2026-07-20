"""End-to-end exercise of the Phase-1 data path on synthetic GTD data."""

from engine.leakage import BANNED_FEATURES
from training.features import build_feature_matrix
from training.report import build_report
from training.split import temporal_split


def test_report_runs(synthetic_gtd, tmp_path):
    r = build_report(synthetic_gtd, tmp_path / "syn.csv")
    assert r["rows"] == len(synthetic_gtd)
    assert 0.0 <= r["target"]["positive_rate"] <= 1.0
    assert r["target"]["labelable_rows"] < r["rows"]  # some rows are unlabelable


def test_features_are_leakage_safe(synthetic_gtd):
    fm = build_feature_matrix(synthetic_gtd)
    # No outcome field survived into X.
    assert not (set(fm.X.columns) & BANNED_FEATURES)
    assert "nkill" not in fm.X.columns
    assert "success" not in fm.X.columns
    # y aligns with X and is binary.
    assert len(fm.X) == len(fm.y)
    assert set(fm.y.unique()) <= {0, 1}
    # GTD "unknown = 0" months became NaN.
    assert fm.X["imonth"].isna().any()


def test_temporal_split_is_ordered(synthetic_gtd):
    fm = build_feature_matrix(synthetic_gtd)
    s = temporal_split(fm.year)
    assert len(s.train) and len(s.test)
    # No index leaks across splits.
    assert set(s.train) & set(s.test) == set()
    assert fm.year.loc[s.train].max() < fm.year.loc[s.test].min()
