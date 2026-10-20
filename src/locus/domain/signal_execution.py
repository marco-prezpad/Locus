"""
signal_execution.py

The signal-to-execution contract: when features are computed, when the
signal becomes available, and when the earliest possible execution
happens. The one-bar-delay invariant is enforced at construction time.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SignalExecutionContract:
    """The contract governing signal availability and execution timing.

    The invariant is hard: `earliest_execution_time` must be a strictly
    later bar than `feature_compute_time`, and intermediate signals must
    not drive execution in the MVP.
    """

    feature_compute_time: Literal["bar_close_T"]
    signal_availability_time: Literal["bar_close_T"]
    earliest_execution_time: Literal["bar_open_T+1"]
    rebalance_cadence_hours: int
    rebalance_anchor: str
    holding_period_hours: int
    signal_used_at_rebalance: Literal["close_T-1"]
    intermediate_signals_used_for_execution: bool
    hysteresis_bps: float
    execution_price: Literal["bar_open_T+1"]

    def __post_init__(self) -> None:
        if self.earliest_execution_time == self.feature_compute_time:
            raise ValueError(
                "earliest_execution_time must be a later bar than "
                "feature_compute_time (one-bar-delay invariant)"
            )
        if self.intermediate_signals_used_for_execution:
            raise ValueError("intermediate signals must not drive execution in the MVP")
        if self.rebalance_cadence_hours <= 0:
            raise ValueError("rebalance_cadence_hours must be positive")
        if self.holding_period_hours <= 0:
            raise ValueError("holding_period_hours must be positive")
        if self.hysteresis_bps < 0:
            raise ValueError("hysteresis_bps must be non-negative")
        if not self.rebalance_anchor:
            raise ValueError("rebalance_anchor must be non-empty")
