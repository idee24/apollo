"""GTD ingestion / quality report.

Produces a JSON + Markdown report on the pinned GTD file: row counts, date
coverage, collection-era distribution, target base rate, missingness, duplicates,
casualty-field consistency, and availability of the Model-A feature columns.

Run once the GTD file is in place:

    python -m training.report

Writes reports/gtd_ingestion_report.{json,md}. The report contains only aggregate
statistics (safe to commit) — never raw incident records.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from engine.schema import (
    FEATURE_COLUMNS,
    MISSING_YEARS,
    TARGET_SOURCE,
    collection_era,
)
from training.config import REPORTS_DIR
from training.data import load_gtd, resolve_gtd_path


def _missingness(df: pd.DataFrame, top: int = 25) -> list[dict]:
    pct = (df.isna().sum() * 100 / len(df)).sort_values(ascending=False)
    return [{"column": c, "percent_missing": round(float(v), 2)} for c, v in pct.head(top).items()]


def _casualty_consistency(df: pd.DataFrame) -> dict:
    checks: dict[str, int] = {}

    def col(name):
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns else None

    nkill, nkillus, nkillter = col("nkill"), col("nkillus"), col("nkillter")
    if nkill is not None:
        checks["nkill_negative"] = int((nkill < 0).sum())
        checks["nkill_non_integer"] = int(((nkill % 1) != 0).sum())
    if nkill is not None and nkillter is not None:
        checks["nkillter_gt_nkill"] = int((nkillter > nkill).sum())
    if nkill is not None and nkillus is not None:
        checks["nkillus_gt_nkill"] = int((nkillus > nkill).sum())
    return checks


def build_report(df: pd.DataFrame, source_path: Path) -> dict:
    years = pd.to_numeric(df.get("iyear"), errors="coerce").dropna().astype(int)
    present = sorted(years.unique().tolist())
    full_range = list(range(min(present), max(present) + 1)) if present else []
    missing = [y for y in full_range if y not in present]

    era_counts = years.map(collection_era).value_counts().to_dict()

    labelable = df[df[TARGET_SOURCE].notna()] if TARGET_SOURCE in df.columns else df.iloc[0:0]
    pos_rate = float((labelable[TARGET_SOURCE] > 0).mean()) if len(labelable) else None

    feat_availability = []
    for c in FEATURE_COLUMNS:
        present_c = c in df.columns
        feat_availability.append({
            "column": c,
            "present": present_c,
            "percent_missing": round(float(df[c].isna().mean() * 100), 2) if present_c else None,
        })

    dup_full = int(df.duplicated().sum())
    dup_eventid = int(df.duplicated(subset=["eventid"]).sum()) if "eventid" in df.columns else None

    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_file": source_path.name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "year_min": int(min(present)) if present else None,
        "year_max": int(max(present)) if present else None,
        "years_missing": missing,
        "expected_missing_years_present_ok": all(y in missing for y in MISSING_YEARS)
        if present else None,
        "collection_era_counts": {k: int(v) for k, v in era_counts.items()},
        "target": {
            "source_column": TARGET_SOURCE,
            "labelable_rows": int(len(labelable)),
            "unlabelable_rows": int(len(df) - len(labelable)),
            "positive_rate": round(pos_rate, 4) if pos_rate is not None else None,
        },
        "duplicates": {"full_row": dup_full, "by_eventid": dup_eventid},
        "casualty_consistency": _casualty_consistency(df),
        "coordinates": {
            "latitude_pct_missing": round(float(df["latitude"].isna().mean() * 100), 2)
            if "latitude" in df.columns else None,
            "longitude_pct_missing": round(float(df["longitude"].isna().mean() * 100), 2)
            if "longitude" in df.columns else None,
        },
        "feature_availability": feat_availability,
        "top_missing_columns": _missingness(df),
    }


def render_markdown(r: dict) -> str:
    lines = [
        "# GTD Ingestion Report",
        "",
        f"- **Generated:** {r['generated_utc']}",
        f"- **Source:** `{r['source_file']}`",
        f"- **Rows:** {r['rows']:,}  **Columns:** {r['columns']}",
        f"- **Year range:** {r['year_min']}–{r['year_max']}",
        f"- **Years missing in range:** {r['years_missing'] or 'none'}",
        "",
        "## Target (`death = nkill > 0`)",
        f"- Labelable rows: {r['target']['labelable_rows']:,}",
        f"- Unlabelable (nkill missing): {r['target']['unlabelable_rows']:,}",
        f"- **Positive rate:** {r['target']['positive_rate']}",
        "",
        "## Collection era distribution",
    ]
    for era, n in r["collection_era_counts"].items():
        lines.append(f"- {era}: {n:,}")
    lines += [
        "",
        "## Data quality",
        f"- Duplicate full rows: {r['duplicates']['full_row']:,}",
        f"- Duplicate eventids: {r['duplicates']['by_eventid']}",
        f"- Casualty consistency: {r['casualty_consistency']}",
        f"- Coordinates missing: lat {r['coordinates']['latitude_pct_missing']}%, "
        f"lon {r['coordinates']['longitude_pct_missing']}%",
        "",
        "## Model-A feature availability",
        "| column | present | % missing |",
        "|---|---|---|",
    ]
    for f in r["feature_availability"]:
        lines.append(f"| {f['column']} | {f['present']} | {f['percent_missing']} |")
    lines += [
        "",
        "## Top missing columns (all)",
        "| column | % missing |",
        "|---|---|",
    ]
    for m in r["top_missing_columns"]:
        lines.append(f"| {m['column']} | {m['percent_missing']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    path = resolve_gtd_path()
    df = load_gtd(path)
    report = build_report(df, path)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "gtd_ingestion_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "gtd_ingestion_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(f"Wrote reports/gtd_ingestion_report.md and .json  ({report['rows']:,} rows)")
    print(f"Positive rate: {report['target']['positive_rate']}  "
          f"| eras: {report['collection_era_counts']}")


if __name__ == "__main__":
    main()
