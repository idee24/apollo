"""Tripwire tests for the leakage guard. These must never be weakened without a
corresponding, justified change to docs/prediction_time_contract.md."""

import pytest

from engine.leakage import (
    BANNED_FEATURES,
    LeakageError,
    assert_leakage_safe,
    find_leaks,
)


def test_core_fatality_fields_are_banned():
    for col in ("nkill", "nkillus", "nkillter", "nwound", "nwoundus", "nwoundte"):
        assert col in BANNED_FEATURES


def test_success_and_outcome_fields_are_banned():
    for col in ("success", "propextent", "hostkidoutcome", "nreleased"):
        assert col in BANNED_FEATURES


def test_clean_feature_set_passes():
    safe = ["iyear", "imonth", "country", "attacktype1", "weapsubtype1", "targtype1"]
    assert find_leaks(safe) == []
    assert_leakage_safe(safe)  # should not raise


def test_leaky_feature_set_is_rejected():
    leaky = ["iyear", "country", "nkill", "attacktype1"]
    assert find_leaks(leaky) == ["nkill"]
    with pytest.raises(LeakageError):
        assert_leakage_safe(leaky)
