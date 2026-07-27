"""Scenario-sweep tests (Phase 5). Verifies the mechanics and the non-forecast
framing. Synthetic model only."""

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.calibration import CalibratedClassifierCV

from api import scenario as scenario_service
from training.features import build_feature_matrix
from training.split import temporal_split
from training.train import build_pipeline


@pytest.fixture
def bundle(synthetic_gtd):
    fm = build_feature_matrix(synthetic_gtd)
    splits = temporal_split(fm.year)
    num = [c for c in fm.X.columns if c in ("iyear", "latitude", "longitude")]
    cat = [c for c in fm.X.columns if c not in num]
    model = CalibratedClassifierCV(build_pipeline(num, cat), method="isotonic", cv=3)
    model.fit(fm.X.loc[splits.train], fm.y.loc[splits.train])
    return {"model": model, "numeric": num, "categorical": cat}


def test_build_points_year_and_month():
    yr = scenario_service.build_points(2010, 2012, "year", month=None)
    assert [p["date"] for p in yr] == ["2010", "2011", "2012"]
    mo = scenario_service.build_points(2010, 2011, "month", month=None)
    assert len(mo) == 24
    assert mo[0]["date"] == "2010-01" and mo[-1]["date"] == "2011-12"


def test_build_points_rejects_bad_range_and_oversize():
    with pytest.raises(ValueError):
        scenario_service.build_points(2015, 2010, "year", None)
    with pytest.raises(ValueError):
        scenario_service.build_points(1970, 2100, "month", None)  # > MAX_POINTS


def test_sweep_varies_only_date(bundle):
    base = {"attacktype1": 3, "weaptype1": 6, "targtype1": 14}
    points = scenario_service.build_points(2005, 2015, "year", month=6)
    out = scenario_service.sweep(bundle, base, points)
    assert len(out) == 11
    for row in out:
        assert 0.0 <= row["probability"] <= 1.0
        assert row["uncertainty_low"] <= row["probability"] <= row["uncertainty_high"]
    assert [r["date"] for r in out] == [str(y) for y in range(2005, 2016)]


@pytest.fixture
def client(bundle, tmp_path, monkeypatch):
    p = tmp_path / "model_A_test.joblib"
    joblib.dump(bundle, p)
    monkeypatch.setenv("APOLLO_MODEL_PATH", str(p))
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_scenario_endpoint_is_labelled_non_forecast(client):
    r = client.post("/v1/scenario", json={
        "start_year": 2000, "end_year": 2010, "by": "year", "month": 6,
        "attacktype1": 3, "weaptype1": 6, "targtype1": 14,
    })
    assert r.status_code == 200
    b = r.json()
    assert len(b["points"]) == 11
    assert "NOT A FORECAST" in b["disclaimer"]
    assert b["scenario"] and b["by"] == "year"


def test_scenario_endpoint_rejects_date_and_outcome_fields(client):
    # iyear/imonth are swept, not accepted; nkill is a banned outcome field.
    assert client.post("/v1/scenario", json={
        "start_year": 2000, "end_year": 2005, "iyear": 2003}).status_code == 422
    assert client.post("/v1/scenario", json={
        "start_year": 2000, "end_year": 2005, "nkill": 2}).status_code == 422


def test_scenario_endpoint_rejects_bad_range(client):
    assert client.post("/v1/scenario", json={
        "start_year": 2010, "end_year": 2000}).status_code == 422
