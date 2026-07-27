"""Canonical place-name normalisation, shared by training and the API.

Training builds the location-catalog keys with :func:`normalize_location`; the
forecast service resolves user input with the *same* function. If the two ever
diverged, every text location would silently fail to resolve ("Unknown
location") while in-process tests stayed green. Keeping the single transform
here in the shared engine layer — alongside the other cross-cutting contracts
(``codes``, ``leakage``, ``schema``) — makes that drift impossible.
"""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_location(value: str) -> str:
    """Case-fold, collapse non-alphanumeric runs to single spaces, and trim.

    Examples: ``"Alaska" -> "alaska"``, ``"country:217" -> "country 217"``.
    Non-ASCII letters are dropped by the ``[^a-z0-9]`` class, which is the
    long-standing behaviour the catalog keys were built against — do not
    "improve" it without rebuilding every artifact's catalog to match.
    """
    return _NON_ALNUM.sub(" ", value.casefold()).strip()
