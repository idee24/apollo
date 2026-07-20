"""API configuration, read lazily from the environment so tests can override it.

Env vars:
  APOLLO_MODEL_PATH          explicit path to a model artifact (else newest in models/)
  APOLLO_API_KEY             if set, requests must send a matching X-API-Key header
  APOLLO_RATE_LIMIT_PER_MIN  per-identity request cap (default 120; <=0 disables)
"""

from __future__ import annotations

import os


def model_path_override() -> str | None:
    return os.getenv("APOLLO_MODEL_PATH")


def api_key() -> str | None:
    return os.getenv("APOLLO_API_KEY")


def rate_limit_per_min() -> int:
    try:
        return int(os.getenv("APOLLO_RATE_LIMIT_PER_MIN", "120"))
    except ValueError:
        return 120
