"""Load the pinned GTD file and verify it against the registry."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml

from training.config import GTD_ENCODING, RAW_DIR, REGISTRY_PATH


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def registry_entry(name: str = "gtd") -> dict:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["datasets"][name]


def resolve_gtd_path() -> Path:
    """Find the GTD file: registry `local_path` if set, else newest CSV under data/raw/gtd/."""
    entry = registry_entry("gtd")
    local = entry.get("local_path")
    if local and str(local) != "TBD":
        p = Path(local)
        return p if p.is_absolute() else (REGISTRY_PATH.parent.parent / p)
    candidates = sorted(
        (RAW_DIR / "gtd").glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(
            "No GTD file found. Obtain it via the request form and place it under "
            "data/raw/gtd/ (see data/raw/README.md), then set local_path/sha256 in registry.yaml."
        )
    return candidates[0]


def load_gtd(path: Path | None = None, *, verify_hash: bool = True) -> pd.DataFrame:
    """Load GTD as a DataFrame, optionally checking its hash against the registry."""
    path = path or resolve_gtd_path()
    if verify_hash:
        expected = registry_entry("gtd").get("sha256")
        if expected and str(expected) != "TBD":
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(
                    f"GTD hash mismatch for {path.name}:\n"
                    f" expected {expected}\n actual   {actual}\n"
                    "The pinned data does not match the registry. Refusing to proceed."
                )
        else:
            print(f"[warn] registry sha256 is unset; skipping hash check for {path.name}. "
                  "Record the hash in registry.yaml to pin this data.")
    return pd.read_csv(path, encoding=GTD_ENCODING, low_memory=False)
