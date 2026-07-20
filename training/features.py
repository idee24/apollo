"""Build the leakage-safe Model-A feature matrix from raw GTD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.leakage import assert_leakage_safe
from engine.schema import (
    TARGET_SOURCE,
    UNKNOWN_AS_ZERO,
    available_feature_columns,
    collection_era,
)


@dataclass
class FeatureMatrix:
    X: pd.DataFrame
    y: pd.Series
    year: pd.Series   # iyear, aligned to X — used for the temporal split
    era: pd.Series    # collection era, aligned to X — used for stratified audits


def build_feature_matrix(df: pd.DataFrame) -> FeatureMatrix:
    """Return leakage-safe (X, y, year, era).

    - Target: death = (nkill > 0), only where nkill is present.
    - Features: PRE_EVENT + SCENARIO columns present in df (see schema).
    - GTD "unknown = 0" month/day values are converted to NaN for the imputer.
    - Fails loudly if any banned outcome field is in X.
    """
    if TARGET_SOURCE not in df.columns:
        raise KeyError(f"GTD is missing '{TARGET_SOURCE}'.")

    labelable = df[df[TARGET_SOURCE].notna()].copy()
    y = (labelable[TARGET_SOURCE] > 0).astype(int)

    cols = available_feature_columns(labelable)
    assert_leakage_safe(cols)  # tripwire
    X = labelable[cols].copy()

    for c in UNKNOWN_AS_ZERO:
        if c in X.columns:
            X[c] = X[c].replace(0, np.nan)

    year = pd.to_numeric(labelable["iyear"], errors="coerce").astype("Int64")
    era = year.map(lambda v: collection_era(int(v)) if pd.notna(v) else "unknown")

    # Final guard on the exact frame handed to the model.
    assert_leakage_safe(X.columns)
    return FeatureMatrix(X=X, y=y, year=year, era=era)
