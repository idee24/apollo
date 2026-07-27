"""API tests. A small model is trained on synthetic data and pointed at via
APOLLO_MODEL_PATH, so these run in CI without the real (non-redistributable) GTD."""

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.calibration import CalibratedClassifierCV

from training.features import build_feature_matrix
from training.split import temporal_split
from training.train import build_pipeline


@pytest.fixture
def artifact_path(synthetic_gtd, tmp_path):
    fm = build_feature_matrix(synthetic_gtd)
    splits = temporal_split(fm.year)
    num = [c for c in fm.X.columns if c in ("iyear", "latitude", "longitude")]
    cat = [c for c in fm.X.columns if c not in num]
    model = CalibratedClassifierCV(build_pipeline(num, cat), method="isotonic", cv=3)
    model.fit(fm.X.loc[splits.train], fm.y.loc[splits.train])
    p = tmp_path / "model_A_test.joblib"
    joblib.dump({
        "model": model,
        "numeric": num,
        "categorical": cat,
        "forecast_reference": fm.X.loc[splits.train],
        "location_catalog": {"alaska": {"label": "Alaska", "country": 4, "region": 1}},
    }, p)
    return p


@pytest.fixture
def client(artifact_path, monkeypatch):
    monkeypatch.setenv("APOLLO_MODEL_PATH", str(artifact_path))
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_version"]


def test_models_lists_only_safe_feature_columns(client):
    active = client.get("/v1/models").json()["active"]
    assert active["version"]
    cols = active["feature_columns"]
    assert "attacktype1" in cols
    assert "nkill" not in cols and "success" not in cols  # no outcome fields exposed


def test_predict_returns_calibrated_probability(client):
    r = client.post("/v1/predict", json={
        "iyear": 2020, "country": 4, "attacktype1": 3,
        "weaptype1": 6, "weapsubtype1": 16, "targtype1": 14,
    })
    assert r.status_code == 200
    b = r.json()
    assert 0.0 <= b["probability"] <= 1.0
    assert b["uncertainty_low"] <= b["probability"] <= b["uncertainty_high"]
    assert b["model_version"] and b["disclaimer"]
    assert "fatality" in b["target"]


def test_predict_rejects_banned_outcome_field(client):
    # Sending an outcome field (nkill) must be refused at the boundary.
    r = client.post("/v1/predict", json={"iyear": 2020, "nkill": 5})
    assert r.status_code == 422


def test_forecast_endpoint_accepts_only_location_and_year(client):
    r = client.post("/v1/forecast", json={"year": 2050, "location": "Alaska"})
    assert r.status_code == 200
    body = r.json()
    assert body["year"] == 2050 and body["location"] == "Alaska"
    assert body["scenarios_evaluated"] > 0
    assert 0 <= body["probability"] <= 1
    assert body["model_version"]

    # Scenario features are intentionally not part of this interface.
    rejected = client.post(
        "/v1/forecast",
        json={"year": 2050, "location": "Alaska", "attacktype1": 3},
    )
    assert rejected.status_code == 422


def test_predict_503_when_no_model(monkeypatch):
    monkeypatch.setenv("APOLLO_MODEL_PATH", "does_not_exist.joblib")
    from api.main import app
    with TestClient(app) as c:
        assert c.post("/v1/predict", json={"iyear": 2020}).status_code == 503
        assert c.get("/health").json()["model_loaded"] is False
