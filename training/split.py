"""Temporal train/val/test split (no random splitting for headline numbers)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from training.config import TRAIN_END_YEAR, VAL_END_YEAR


@dataclass
class Splits:
    train: pd.Index
    val: pd.Index
    test: pd.Index

    def summary(self, y: pd.Series) -> dict:
        def part(idx):
            s = y.loc[idx]
            return {
                "n": int(len(s)),
                "positive_rate": round(float(s.mean()), 4) if len(s) else None,
            }
        return {"train": part(self.train), "val": part(self.val), "test": part(self.test)}


def temporal_split(
    year: pd.Series,
    *,
    train_end: int = TRAIN_END_YEAR,
    val_end: int = VAL_END_YEAR,
) -> Splits:
    """Split indices by incident year: train <= train_end < val <= val_end < test."""
    y = pd.to_numeric(year, errors="coerce")
    train = year.index[y <= train_end]
    val = year.index[(y > train_end) & (y <= val_end)]
    test = year.index[y > val_end]
    return Splits(train=train, val=val, test=test)
