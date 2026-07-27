"""Tests for the location/year marginal forecast."""

import pandas as pd
import pytest

from api.forecast import forecast
from training.features import build_feature_matrix
from training.train import _build_forecast_reference, _build_location_catalog, build_pipeline


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


def test_forecast_reports_local_population_basis(forecast_bundle):
    # Region 1 has plenty of reference rows, so the mix is marginalised locally.
    out = forecast(forecast_bundle, 2050, "region:1")
    assert out["population_basis"] == "region:1"


def test_forecast_falls_back_to_global_for_sparse_geography(forecast_bundle):
    # A geography with no reference rows must fall back to the global mix, and say so.
    out = forecast(forecast_bundle, 2050, "region:999")
    assert out["population_basis"] == "global"
    assert out["scenarios_evaluated"] > 0


def test_forecast_positive_points_are_distinct(forecast_bundle):
    out = forecast(forecast_bundle, 2050, "Alaska")
    signatures = [tuple(sorted(p["features"].items())) for p in out["positive_points"]]
    assert len(signatures) == len(set(signatures))


def test_forecast_rejects_unknown_location(forecast_bundle):
    with pytest.raises(ValueError, match="Unknown location"):
        forecast(forecast_bundle, 2050, "Atlantis")


def test_forecast_rejects_ambiguous_place_name(forecast_bundle):
    forecast_bundle["location_catalog"]["georgia"] = [
        {"label": "Georgia", "country": 74, "region": 7},
        {"label": "Georgia", "country": 217, "region": 1},
    ]
    with pytest.raises(ValueError, match="Ambiguous location"):
        forecast(forecast_bundle, 2050, "Georgia")


def test_reference_sample_preserves_rare_geographies():
    frame = pd.DataFrame({
        "country": [1] * 100 + [2] * 2,
        "attacktype1": range(102),
    })
    sampled = _build_forecast_reference(frame, limit=27)
    assert len(sampled) == 27
    assert set(sampled["country"]) == {1, 2}


def test_location_catalog_uses_all_training_rows_and_tracks_ambiguity():
    frame = pd.DataFrame({
        "country_txt": ["One", "One", "Two", "Two"],
        "provstate": ["Shared", "Shared", "Shared", "Shared"],
        "country": [1, 1, 2, 2],
        "region": [3, 3, 4, 4],
        "latitude": [1.0, 1.2, 2.0, 2.2],
        "longitude": [3.0, 3.2, 4.0, 4.2],
    })
    catalog = _build_location_catalog(frame, frame.index)
    assert len(catalog["shared"]) == 2
    assert catalog["one"][0]["country"] == 1
