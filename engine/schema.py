"""GTD schema contract for Apollo Model A.

Defines the target, the *allowed* feature columns (PRE_EVENT + SCENARIO from
``docs/prediction_time_contract.md``), and GTD collection eras. This is the
positive counterpart to ``engine/leakage.py`` (which defines what is banned).

Feature choices here are a deliberately modest baseline. Enrichment earns its
place only through out-of-time improvement (STRATEGY.md §9).
"""

from __future__ import annotations

import pandas as pd

from engine.leakage import assert_leakage_safe

# --- Target -----------------------------------------------------------------
TARGET_COL = "death"
TARGET_SOURCE = "nkill"  # death := (nkill > 0); nkill itself is BANNED as a feature.

# --- Date fields (GTD uses 0 to mean "unknown" for month/day) ----------------
DATE_FIELDS = ["iyear", "imonth", "iday"]
UNKNOWN_AS_ZERO = ["imonth", "iday"]  # 0 -> missing

# --- Allowed feature candidates (PRE_EVENT + SCENARIO) -----------------------
# Treated as real numbers:
NUMERIC_FEATURES = ["iyear", "latitude", "longitude"]

# Treated as categorical codes (one-hot encoded). GTD stores these as integer
# codes; we keep the code columns and drop the parallel *_txt label columns.
CATEGORICAL_FEATURES = [
    "imonth",        # seasonality
    "country",       # PRE_EVENT
    "region",        # PRE_EVENT
    "specificity",   # geocoding precision (PRE_EVENT)
    "vicinity",      # in/near city (PRE_EVENT)
    "attacktype1",   # SCENARIO
    "targtype1",     # SCENARIO
    "targsubtype1",  # SCENARIO
    "natlty1",       # SCENARIO (target nationality)
    "weaptype1",     # SCENARIO
    "weapsubtype1",  # SCENARIO
    "suicide",       # SCENARIO (attack modality)
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# NOTE: intentionally excluded from the baseline (see contract doc):
#   provstate/city (very high cardinality), multiple (often coded later),
#   gname/nperps/claimed/INT_* (AMBIGUOUS or POST_OUTCOME).

# --- GTD collection eras (approximate; verify against the codebook) ----------
# Reporting intensity is NOT constant across these regimes.
COLLECTION_ERAS: list[tuple[int, int, str]] = [
    (1970, 1997, "PGIS"),
    (1998, 2007, "CETIS"),
    (2008, 2011, "ISVG"),
    (2012, 9999, "START"),
]
MISSING_YEARS = [1993]  # GTD 1993 data was lost.


def collection_era(year: int) -> str:
    for lo, hi, name in COLLECTION_ERAS:
        if lo <= year <= hi:
            return name
    return "unknown"


def make_target(df: pd.DataFrame) -> pd.Series:
    """death := (nkill > 0), defined only where nkill is present."""
    if TARGET_SOURCE not in df.columns:
        raise KeyError(f"GTD is missing '{TARGET_SOURCE}', required to build the target.")
    labelable = df[df[TARGET_SOURCE].notna()]
    return (labelable[TARGET_SOURCE] > 0).astype(int)


def available_feature_columns(df: pd.DataFrame) -> list[str]:
    """FEATURE_COLUMNS actually present in this dataframe (order preserved)."""
    return [c for c in FEATURE_COLUMNS if c in df.columns]


def split_feature_types(columns: list[str]) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) subsets of ``columns`` per the contract."""
    num = [c for c in columns if c in NUMERIC_FEATURES]
    cat = [c for c in columns if c in CATEGORICAL_FEATURES]
    # Fail loudly if an outcome field ever reaches the feature set.
    assert_leakage_safe(columns)
    return num, cat
