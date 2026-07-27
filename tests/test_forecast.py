"""Tests for the location/year marginal forecast."""

import pandas as pd
import pytest

from api.forecast import forecast
from training.features import build_feature_matrix
from training.train import build_pipeline


@pytest.fixture
def forecast_bundle(synthetic_gtd):
    fm = build_feature_matrix(synthetic_gtd)
    num = [c for c in fm.X if c in ("iyear", "latitude", "longitude")]
    cat = [c for c in fm.X if c not in num]
    model = build_pipeline(num, cat).fit(fm.X, fm.y)
    return {
        "model": model,
        "numeric": num,
        "categorical": cat,
        "forecast_reference": fm.X,
        "location_catalog": {
            "alaska": {
                "label": "Alaska",
                "country": 4,
                "region": 1,
                "latitude": 64.2,
                "longitude": -152.5,
            }
        },
    }


def test_forecast_only_needs_year_and_location(forecast_bundle):
    out = forecast(forecast_bundle, 2050, "Alaska")
    assert out["year"] == 2050 and out["location"] == "Alaska"
    assert 0 <= out["probability"] <= 1
    assert out["scenarios_evaluated"] > 0
    assert out["distribution"]["p05"] <= out["distribution"]["p95"]
    assert len(out["positive_points"]) == 5
    assert "extrapolation" in out["disclaimer"].lower()


def test_forecast_varies_nuisance_features(forecast_bundle):
    ref = pd.DataFrame(forecast_bundle["forecast_reference"])
    assert ref["attacktype1"].nunique() > 1
    out = forecast(forecast_bundle, 2020, "country:4")
    assert out["method"].startswith("Empirical marginalisation")


def test_forecast_rejects_unknown_location(forecast_bundle):
    with pytest.raises(ValueError, match="Unknown location"):
        forecast(forecast_bundle, 2050, "Atlantis")
