"""Phase 1 configuration: paths, temporal split boundaries, encoding."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REGISTRY_PATH = DATA_DIR / "registry.yaml"
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"

# The official GTD CSV ships as Latin-1. Apollo's pinned file, however, is a
# UTF-8 CSV derived once from the official .xlsx releases (main 1970-2020 +
# 2021 H1 supplement) and pinned by SHA-256 in the registry. See data/raw/README.md.
GTD_ENCODING = "utf-8"

# Temporal split (STRATEGY.md §9): no random split for the headline number.
# The latest period is an untouched test set.
TRAIN_END_YEAR = 2015   # train: iyear <= 2015
VAL_END_YEAR = 2018     # val:   2016..2018
# test: iyear >= 2019
