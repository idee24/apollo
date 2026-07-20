"""Canonical leakage controls for Apollo.

This module is the single source of truth for which GTD fields must NEVER be used
as predictors of the fatality target. `docs/prediction_time_contract.md` is the
human-readable companion; keep the two in sync.

The fatality target is ``death = (total fatalities > 0)``. Therefore every
fatality-, casualty-, and outcome-derived field is either the label's basis or a
sibling of it, and using any of them as a feature leaks the answer. This exact
mistake inflated draft 1's reported accuracy.
"""

from __future__ import annotations

from collections.abc import Iterable

# --- The label basis: fatality/casualty counts. Never predictors. ---
FATALITY_FIELDS: frozenset[str] = frozenset(
    {"nkill", "nkillus", "nkillter", "nwound", "nwoundus", "nwoundte"}
)

# --- Other fields known only AFTER the incident resolves. Never predictors. ---
POST_OUTCOME_FIELDS: frozenset[str] = frozenset(
    {
        "success",
        "property", "propextent", "propextent_txt", "propvalue", "propcomment",
        "ishostkid", "nhostkid", "nhostkidus", "nhours", "ndays", "divert",
        "kidhijcountry", "ransom", "ransomamt", "ransomamtus", "ransompaid",
        "ransompaidus", "ransomnote", "hostkidoutcome", "hostkidoutcome_txt",
        "nreleased",
        # source/coding metadata — not real-world predictors
        "addnotes", "scite1", "scite2", "scite3", "dbsource", "related",
    }
)

# The full banned set enforced by the guard.
BANNED_FEATURES: frozenset[str] = FATALITY_FIELDS | POST_OUTCOME_FIELDS

# --- Ambiguous fields: excluded by default (see the contract doc). Promoting one
# to a predictor requires a written justification for a specific prediction moment. ---
AMBIGUOUS_FIELDS: frozenset[str] = frozenset(
    {
        "gname", "gsubname", "guncertain1", "individual",
        "nperps", "nperpcap",
        "claimed", "claimmode", "claimmode_txt",
        "weapdetail", "summary",
        "doubtterr", "crit1", "crit2", "crit3",
        "INT_LOG", "INT_IDEO", "INT_MISC", "INT_ANY",
    }
)


class LeakageError(ValueError):
    """Raised when a banned (outcome-derived) field appears in a feature set."""


def find_leaks(columns: Iterable[str]) -> list[str]:
    """Return any banned columns present in ``columns`` (sorted, deduplicated)."""
    cols = set(columns)
    return sorted(cols & BANNED_FEATURES)


def assert_leakage_safe(columns: Iterable[str]) -> None:
    """Raise :class:`LeakageError` if any banned field is present.

    Call this on the training feature matrix (e.g. ``assert_leakage_safe(X.columns)``)
    so a leak fails loudly at build time rather than silently inflating metrics.
    """
    leaks = find_leaks(columns)
    if leaks:
        raise LeakageError(
            "Outcome-derived field(s) present in features: "
            f"{leaks}. These are banned by docs/prediction_time_contract.md. "
            "Fatality/casualty/outcome fields can never predict the fatality target."
        )
