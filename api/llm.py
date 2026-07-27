"""Provider-agnostic LLM client for Model C (Phase 4).

Model C only *phrases* an explanation; it never produces or changes a number
(that is Model A's job — see api/explain.py). The client is therefore optional:
if no provider is configured, explanations fall back to a deterministic template.

Configuration is server-side only, via environment variables — keys never reach a
client and are never logged or echoed in a response:

    APOLLO_LLM_PROVIDER   "none" (default) | "anthropic"
    APOLLO_LLM_API_KEY    provider key (server-side secret)
    APOLLO_LLM_MODEL      model id (default: a current Claude model)

To add a provider, implement ``LLMClient.generate`` and register it in ``get_llm``.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def generate(self, system: str, user: str) -> str | None:
        """Return generated text, or None on any failure (caller falls back)."""
        ...


class AnthropicLLM:
    """Thin Anthropic Messages API adapter (via httpx). Optional dependency."""

    def __init__(self, api_key: str, model: str):
        self._key = api_key
        self.name = f"anthropic:{model}"
        self._model = model

    def generate(self, system: str, user: str) -> str | None:
        try:
            import httpx  # local import: only needed when a provider is active
        except ImportError:
            return None
        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": 400,
                    "temperature": 0.2,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
                timeout=20.0,
            )
            resp.raise_for_status()
            blocks = resp.json().get("content", [])
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            return text.strip() or None
        except (httpx.HTTPError, ValueError, KeyError):
            # Never surface provider errors (or keys) to the caller; degrade to template.
            return None


def get_llm() -> LLMClient | None:
    """Build the configured client, or None (template-only mode)."""
    provider = os.getenv("APOLLO_LLM_PROVIDER", "none").lower()
    if provider in ("", "none"):
        return None
    key = os.getenv("APOLLO_LLM_API_KEY")
    if not key:
        return None
    model = os.getenv("APOLLO_LLM_MODEL", "claude-sonnet-5")
    if provider == "anthropic":
        return AnthropicLLM(api_key=key, model=model)
    return None
