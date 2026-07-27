"""Apollo inference API (Phase 2).

    uvicorn api.main:app --reload

Loads the active model artifact once at startup. Serves only derived model
outputs — never raw GTD records (a design choice and a GTD licence requirement).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from api import explain as explain_service
from api import service
from api.knowledge import build_retriever
from api.llm import get_llm
from api.model_registry import load_model
from api.schemas import ExplainResponse, PredictRequest, PredictResponse
from api.security import rate_limit, require_api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()  # None until a model is trained
    app.state.retriever = build_retriever()  # RAG corpus (Apollo docs); None if absent
    app.state.llm = get_llm()  # provider-agnostic; None -> deterministic template
    yield


app = FastAPI(
    title="Apollo Engine API",
    version="0.1.0",
    summary="Conditional lethality estimator — retrospective, probabilistic decision-support.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    m = app.state.model
    return {
        "status": "ok",
        "model_loaded": m is not None,
        "model_version": m["version"] if m else None,
    }


@app.get("/v1/models")
def models() -> dict:
    m = app.state.model
    if not m:
        return {"active": None}
    b = m["bundle"]
    return {
        "active": {
            "version": m["version"],
            "artifact": m["path"].name,
            "metrics": m.get("metrics"),
            "feature_columns": list(b["numeric"]) + list(b["categorical"]),
            "intended_use": "docs/intended_use.md",
            "model_card": "docs/model_card.md",
        }
    }


@app.post(
    "/v1/predict",
    response_model=PredictResponse,
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
def predict(req: PredictRequest) -> PredictResponse:
    m = app.state.model
    if not m:
        raise HTTPException(status_code=503, detail="No model loaded. Train Model A first.")
    out = service.predict(m["bundle"], req.model_dump(exclude_none=True))
    out["model_version"] = m["version"]
    return PredictResponse(**out)


@app.post(
    "/v1/explain",
    response_model=ExplainResponse,
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
def explain(req: PredictRequest) -> ExplainResponse:
    m = app.state.model
    if not m:
        raise HTTPException(status_code=503, detail="No model loaded. Train Model A first.")
    out = explain_service.explain(
        m["bundle"], req.model_dump(exclude_none=True),
        retriever=app.state.retriever, llm=app.state.llm,
    )
    out["model_version"] = m["version"]
    return ExplainResponse(**out)
