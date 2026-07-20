"""Phase 1 configuration: paths, temporal split boundaries, encoding."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REGISTRY_PATH = DATA_DIR / "registry.yaml"
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"

# GTD ships as Latin-1, not UTF-8.
GTD_ENCODING = "ISO-8859-1"

# Temporal split (STRATEGY.md §9): no random split for the headline number.
# The latest period is an untouched test set.
TRAIN_END_YEAR = 2015   # train: iyear <= 2015
VAL_END_YEAR = 2018     # val:   2016..2018
# test: iyear >= 2019
