"""Model C (explanation) tests — retrieval, scenario description, and the
hard rule that narrative can never alter Model A's number. Synthetic only."""

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.calibration import CalibratedClassifierCV

from api import explain as explain_service
from api import service
from api.knowledge import build_retriever
from engine.codes import describe_scenario
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


class _LyingLLM:
    name = "fake:liar"

    def generate(self, system, user):
        return "This scenario is certain to be fatal with probability 0.99."


def test_describe_scenario_maps_codes():
    s = describe_scenario({"attacktype1": 3, "weaptype1": 6, "targtype1": 14,
                           "region": 10, "iyear": 2019, "suicide": 1})
    assert "bombing/explosion" in s.lower()
    assert "explosives" in s.lower()
    assert "private citizens" in s.lower()
    assert "Middle East & North Africa" in s
    assert "2019" in s
    assert "suicide attack" in s.lower()


def test_retriever_returns_grounded_evidence():
    r = build_retriever()
    assert r is not None
    ev = r.retrieve("calibration brier score fairness subgroup", k=4)
    assert ev, "expected at least one relevant chunk"
    assert any(e.source.startswith("model_card.md") for e in ev)
    assert all(0.0 <= e.score <= 1.0 for e in ev)


def test_explanation_template_mode_is_grounded(bundle):
    feats = {"iyear": 2020, "attacktype1": 3, "weaptype1": 6, "targtype1": 14}
    out = explain_service.explain(bundle, feats, retriever=build_retriever(), llm=None)
    assert out["generated_by"] == "template"
    assert out["evidence"]
    assert out["disclaimer"] in out["explanation"]
    # The number in the response equals Model A's number.
    assert out["probability"] == service.predict(bundle, feats)["probability"]


def test_llm_cannot_alter_the_number(bundle):
    feats = {"iyear": 2020, "attacktype1": 2, "weaptype1": 5, "targtype1": 4}
    truth = service.predict(bundle, feats)["probability"]
    out = explain_service.explain(bundle, feats, retriever=build_retriever(), llm=_LyingLLM())
    # Even though the LLM asserted 0.99, the authoritative field is Model A's.
    assert out["probability"] == truth
    assert out["generated_by"] == "fake:liar"
    # The exact probability line and disclaimer are re-anchored by code.
    assert f"{truth:.2f}" in out["explanation"]
    assert out["disclaimer"] in out["explanation"]


@pytest.fixture
def client(bundle, tmp_path, monkeypatch):
    p = tmp_path / "model_A_test.joblib"
    joblib.dump(bundle, p)
    monkeypatch.setenv("APOLLO_MODEL_PATH", str(p))
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_explain_endpoint(client):
    r = client.post("/v1/explain", json={"iyear": 2020, "attacktype1": 3, "weaptype1": 6})
    assert r.status_code == 200
    b = r.json()
    assert 0.0 <= b["probability"] <= 1.0
    assert b["generated_by"] == "template"
    assert b["scenario"] and b["disclaimer"]
    assert isinstance(b["evidence"], list)


def test_explain_endpoint_rejects_banned_field(client):
    r = client.post("/v1/explain", json={"iyear": 2020, "nkill": 3})
    assert r.status_code == 422
