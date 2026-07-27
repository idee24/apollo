"""Responsible-AI audit for Model A (Phase 3).

    python -m training.audit

Loads the pinned GTD, rebuilds the exact temporal split, and on the untouched
test set reports:
  1. per-subgroup fairness (region / country / target nationality), and
  2. a sensitive-field ablation — the same calibrated pipeline trained with vs.
     without the sensitive fields, so the accuracy cost of using them is explicit.

Writes reports/fairness_A.json. The audited model is the same deterministic
pipeline the trainer persists (RandomForest+isotonic, random_state=42), so these
numbers describe the deployed model.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sklearn.calibration import CalibratedClassifierCV

from engine.schema import split_feature_types
from training.config import REPORTS_DIR, SENSITIVE_FIELDS
from training.data import load_gtd, registry_entry
from training.fairness import reliability_curve, subgroup_report
from training.features import build_feature_matrix
from training.metrics import evaluate_proba
from training.split import temporal_split
from training.train import build_pipeline

# GTD label columns used ONLY to name subgroups in the report — never features.
GROUP_LABELS = {"region": "region_txt", "country": "country_txt", "nationality": "natlty1_txt"}


def _fit_calibrated(Xtr, ytr, cols):
    numeric, categorical = split_feature_types(cols)
    base = build_pipeline(numeric, categorical)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(Xtr[cols], ytr)
    return model


def run_audit() -> dict:
    df = load_gtd()
    fm = build_feature_matrix(df)
    splits = temporal_split(fm.year)
    all_cols = list(fm.X.columns)
    Xtr, ytr = fm.X.loc[splits.train], fm.y.loc[splits.train]
    Xte, yte = fm.X.loc[splits.test], fm.y.loc[splits.test]

    # --- Deployed model (with sensitive fields) ------------------------------
    full = _fit_calibrated(Xtr, ytr, all_cols)
    proba = full.predict_proba(Xte[all_cols])[:, 1]
    overall = evaluate_proba(yte, proba)

    # --- Per-subgroup fairness (labels pulled from raw *_txt, not features) --
    subgroups = {}
    for name, txt_col in GROUP_LABELS.items():
        if txt_col not in df.columns:
            continue
        labels = df.loc[Xte.index, txt_col]
        subgroups[name] = subgroup_report(yte.to_numpy(), proba, labels.to_numpy())

    # --- Sensitive-field ablation --------------------------------------------
    ablated_cols = [c for c in all_cols if c not in SENSITIVE_FIELDS]
    ablated = _fit_calibrated(Xtr, ytr, ablated_cols)
    ablated_metrics = evaluate_proba(yte, ablated.predict_proba(Xte[ablated_cols])[:, 1])
    ablation = {
        "sensitive_fields": SENSITIVE_FIELDS,
        "with_sensitive": overall,
        "without_sensitive": ablated_metrics,
        "cost_of_removing": {
            "roc_auc": round((overall["roc_auc"] or 0) - (ablated_metrics["roc_auc"] or 0), 4),
            "brier": round((ablated_metrics["brier"] or 0) - (overall["brier"] or 0), 4),
        },
    }

    result = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "gtd_version": registry_entry("gtd").get("version"),
        "test_split": {"n": int(len(yte)), "positive_rate": round(float(yte.mean()), 4)},
        "overall_metrics": overall,
        "reliability_curve": reliability_curve(yte.to_numpy(), proba),
        "subgroups": subgroups,
        "ablation": ablation,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "fairness_A.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    r = run_audit()
    print(f"Test set: n={r['test_split']['n']} pos_rate={r['test_split']['positive_rate']}")
    om = r["overall_metrics"]
    print(f"Overall: ROC-AUC={om['roc_auc']} Brier={om['brier']}")
    for name, rep in r["subgroups"].items():
        d = rep["disparities"]
        tpr = d.get("tpr") if d else None
        print(f"\n[{name}] {rep['n_groups_reported']} groups reported, "
              f"{rep['suppressed']['n_groups']} suppressed (<{rep['min_size']} rows)")
        if tpr:
            print(f"  TPR range {tpr['range']}  "
                  f"(min {tpr['min']['group']}={tpr['min']['value']}, "
                  f"max {tpr['max']['group']}={tpr['max']['value']})")
        if d and d.get("worst_calibration_gap"):
            w = d["worst_calibration_gap"]
            print(f"  worst calibration gap: {w['group']} = {w['calibration_gap']}")
    ab = r["ablation"]
    print(f"\nAblation (remove {ab['sensitive_fields']}): "
          f"ROC-AUC {ab['with_sensitive']['roc_auc']} -> {ab['without_sensitive']['roc_auc']} "
          f"(cost {ab['cost_of_removing']['roc_auc']}), "
          f"Brier {ab['with_sensitive']['brier']} -> {ab['without_sensitive']['brier']}")


if __name__ == "__main__":
    main()
