"""
test_signal_execution.py

Verifies the one-bar-delay invariant.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import pytest

from locus.domain import SignalExecutionContract


def _contract(**overrides: object) -> SignalExecutionContract:
    defaults = {
        "feature_compute_time": "bar_close_T",
        "signal_availability_time": "bar_close_T",
        "earliest_execution_time": "bar_open_T+1",
        "rebalance_cadence_hours": 24,
        "rebalance_anchor": "00:00 UTC",
        "holding_period_hours": 24,
        "signal_used_at_rebalance": "close_T-1",
        "intermediate_signals_used_for_execution": False,
        "hysteresis_bps": 50.0,
        "execution_price": "bar_open_T+1",
    }
    defaults.update(overrides)
    return SignalExecutionContract(**defaults)  # type: ignore[arg-type]


def test_contract_default_invariants() -> None:
    c = _contract()
    assert c.earliest_execution_time != c.feature_compute_time
    assert c.intermediate_signals_used_for_execution is False


def test_contract_rejects_same_bar_execution() -> None:
    with pytest.raises(ValueError):
        _contract(earliest_execution_time="bar_close_T")


def test_contract_rejects_intermediate_execution() -> None:
    with pytest.raises(ValueError):
        _contract(intermediate_signals_used_for_execution=True)


def test_contract_rejects_non_positive_rebalance() -> None:
    with pytest.raises(ValueError):
        _contract(rebalance_cadence_hours=0)


def test_contract_rejects_negative_hysteresis() -> None:
    with pytest.raises(ValueError):
        _contract(hysteresis_bps=-1.0)


def test_contract_rejects_empty_anchor() -> None:
    with pytest.raises(ValueError):
        _contract(rebalance_anchor="")
