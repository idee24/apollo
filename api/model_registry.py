"""Locate and load the active model artifact (once, at startup)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from api.config import model_path_override
from training.config import MODELS_DIR, REPORTS_DIR


def find_artifact() -> Path | None:
    override = model_path_override()
    if override:
        p = Path(override)
        return p if p.exists() else None
    if not MODELS_DIR.exists():
        return None
    arts = sorted(
        MODELS_DIR.glob("model_A_*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return arts[0] if arts else None


def _metrics_for(artifact_name: str) -> dict | None:
    mfile = REPORTS_DIR / "model_A_metrics.json"
    if not mfile.exists():
        return None
    try:
        data = json.loads(mfile.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("artifact") == artifact_name:
        return {"metrics_test": data.get("model", {}).get("metrics_test"),
                "beats_baselines": data.get("beats_baselines")}
    return None


def load_model() -> dict | None:
    """Return {bundle, path, version, metrics} or None if no artifact is available.

    `bundle` is the dict persisted by training.train: {model, numeric, categorical}.
    """
    path = find_artifact()
    if not path or not path.exists():
        return None
    bundle = joblib.load(path)
    return {
        "bundle": bundle,
        "path": path,
        "version": path.stem,
        "metrics": _metrics_for(path.name),
    }
