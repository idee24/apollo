"""Shared fixtures: a small synthetic GTD-like frame for exercising the pipeline
without the real (non-redistributable) dataset."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_gtd() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 4000
    years = rng.integers(1970, 2022, n)
    attack = rng.integers(1, 10, n)
    weapsub = rng.integers(1, 30, n)
    # A learnable signal: some attack/weapon types are more lethal, plus noise.
    logit = -0.3 + 0.25 * (attack % 4) + 0.15 * (weapsub % 5) + rng.normal(0, 1.0, n)
    p = 1 / (1 + np.exp(-logit))
    death = rng.binomial(1, p)
    nkill = np.where(death == 1, rng.integers(1, 20, n), 0).astype(float)
    # Sprinkle some unlabelable rows.
    nkill[rng.random(n) < 0.05] = np.nan

    return pd.DataFrame({
        "eventid": np.arange(n),
        "iyear": years,
        "imonth": rng.integers(0, 13, n),   # includes 0 = unknown
        "iday": rng.integers(0, 29, n),
        "country": rng.integers(1, 200, n),
        "region": rng.integers(1, 13, n),
        "specificity": rng.integers(1, 6, n),
        "vicinity": rng.integers(0, 2, n),
        "latitude": rng.uniform(-60, 70, n),
        "longitude": rng.uniform(-180, 180, n),
        "attacktype1": attack,
        "targtype1": rng.integers(1, 23, n),
        "targsubtype1": rng.integers(1, 100, n),
        "natlty1": rng.integers(1, 200, n),
        "weaptype1": rng.integers(1, 14, n),
        "weapsubtype1": weapsub,
        "suicide": rng.integers(0, 2, n),
        # outcome fields that MUST NOT become features:
        "nkill": nkill,
        "nwound": rng.integers(0, 30, n).astype(float),
        "success": rng.integers(0, 2, n),
    })
