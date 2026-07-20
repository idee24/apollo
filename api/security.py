"""Minimal, dependency-free API-key auth and in-memory rate limiting."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

from api.config import api_key, rate_limit_per_min

_hits: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    key = api_key()
    if key and x_api_key != key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def rate_limit(request: Request) -> None:
    limit = rate_limit_per_min()
    if limit <= 0:
        return
    ident = request.headers.get("x-api-key") or (request.client.host if request.client else "anon")
    now = time.time()
    with _lock:
        dq = _hits[ident]
        while dq and now - dq[0] > 60:
            dq.popleft()
        if len(dq) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        dq.append(now)
