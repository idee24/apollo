"""Apollo inference API (Phase 2).

    uvicorn api.main:app --reload

Loads the active model artifact once at startup. Serves only derived model
outputs — never raw GTD records (a design choice and a GTD licence requirement).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from api import service
from api.model_registry import load_model
from api.schemas import PredictRequest, PredictResponse
from api.security import rate_limit, require_api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()  # None until a model is trained
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
