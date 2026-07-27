"""Train, calibrate, evaluate and persist Apollo Model A.

    python -m training.train

Pipeline: ColumnTransformer(impute + one-hot for categoricals, impute + scale for
numerics) -> classifier, wrapped in probability calibration. Evaluated on an
untouched temporal test set against the required baselines. Everything (imputers,
encoders, calibration) is fit inside the training data only.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from engine.leakage import assert_leakage_safe
from engine.schema import split_feature_types
from training.baselines import default_baselines
from training.config import MODELS_DIR, REPORTS_DIR
from training.data import load_gtd, registry_entry, resolve_gtd_path
from training.features import build_feature_matrix
from training.metrics import evaluate_proba
from training.split import temporal_split


def build_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
            ]), categorical),
        ],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=4, n_jobs=-1, random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def train_and_evaluate() -> dict:
    path = resolve_gtd_path()
    df = load_gtd(path)
    fm = build_feature_matrix(df)
    assert_leakage_safe(fm.X.columns)  # belt and suspenders

    splits = temporal_split(fm.year)
    numeric, categorical = split_feature_types(list(fm.X.columns))

    Xtr, ytr = fm.X.loc[splits.train], fm.y.loc[splits.train]
    Xte, yte = fm.X.loc[splits.test], fm.y.loc[splits.test]

    # --- Model: calibrated pipeline, calibration fit within training only ---
    base = build_pipeline(numeric, categorical)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(Xtr, ytr)
    model_metrics = evaluate_proba(yte, model.predict_proba(Xte)[:, 1])

    # --- Baselines (must be beaten) ---
    baseline_metrics = {}
    for name, bl in default_baselines(list(fm.X.columns)).items():
        bl.fit(Xtr, ytr)
        baseline_metrics[name] = evaluate_proba(yte, bl.predict_proba(Xte))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    MODELS_DIR.mkdir(exist_ok=True)
    artifact = MODELS_DIR / f"model_A_{stamp}.joblib"
    forecast_reference = Xtr.sample(min(10_000, len(Xtr)), random_state=42)
    location_catalog = _build_location_catalog(df, forecast_reference.index)
    joblib.dump({
        "model": model,
        "numeric": numeric,
        "categorical": categorical,
        "forecast_reference": forecast_reference,
        "location_catalog": location_catalog,
    }, artifact)

    result = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "gtd_version": registry_entry("gtd").get("version"),
        "artifact": artifact.name,
        "split": splits.summary(fm.y),
        "features": {"numeric": numeric, "categorical": categorical},
        "model": {"type": "RandomForest+isotonic-calibration", "metrics_test": model_metrics},
        "baselines": baseline_metrics,
        "beats_baselines": _beats(model_metrics, baseline_metrics),
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "model_A_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _normalise_location(value: str) -> str:
    """Keep training independent of the API package while matching its keys."""
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _build_location_catalog(df, indices) -> dict[str, dict]:
    """Build place aliases without shipping any raw incident or narrative text."""
    available = df.loc[df.index.intersection(indices)]
    catalog: dict[str, dict] = {}
    for text_col in ("country_txt", "provstate", "city"):
        if text_col not in available:
            continue
        for name, group in available.dropna(subset=[text_col]).groupby(text_col):
            key = _normalise_location(str(name))
            if not key or len(group) < 2:
                continue
            entry: dict = {"label": str(name)}
            for code in ("country", "region"):
                if code in group and group[code].notna().any():
                    entry[code] = int(group[code].mode().iloc[0])
            for coord in ("latitude", "longitude"):
                if coord in group and group[coord].notna().any():
                    entry[coord] = float(group[coord].median())
            # Prefer the more specific alias when duplicate names occur.
            catalog[key] = entry
    return catalog


def _beats(model_metrics: dict, baseline_metrics: dict) -> dict:
    """Gate check: does the model beat every baseline on ROC-AUC and Brier?"""
    m_auc, m_brier = model_metrics.get("roc_auc"), model_metrics.get("brier")
    verdict = {}
    for name, bm in baseline_metrics.items():
        auc_ok = (m_auc is not None and bm.get("roc_auc") is not None and m_auc > bm["roc_auc"])
        brier_ok = (m_brier is not None and bm.get("brier") is not None and m_brier < bm["brier"])
        verdict[name] = {"better_auc": bool(auc_ok), "better_brier": bool(brier_ok)}
    verdict["PASS_GATE"] = all(
        v["better_auc"] and v["better_brier"]
        for k, v in verdict.items()
        if k != "PASS_GATE"
    )
    return verdict


def main() -> None:
    r = train_and_evaluate()
    print(f"Artifact: {r['artifact']}")
    print(f"Split: {r['split']}")
    print(f"Model (test): {r['model']['metrics_test']}")
    for name, bm in r["baselines"].items():
        print(f"Baseline {name}: {bm}")
    print(f"Gate - beats all baselines (AUC & Brier): {r['beats_baselines']['PASS_GATE']}")


if __name__ == "__main__":
    main()
