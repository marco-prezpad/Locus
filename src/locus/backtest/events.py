"""
events.py

Event types and fill/report contracts. The engine is a state machine
driven by a timeline of rebalance timestamps; events are data classes
that record what happened, not a pub/sub bus (that's F6.3 for live).

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from locus.evaluation.metrics import PerformanceMetrics

Side = Literal["buy", "sell"]


@dataclass(frozen=True, slots=True)
class FillEvent:
    timestamp: datetime
    asset_id: str
    side: Side
    quantity: float
    fill_price: float
    cost_bps: float
    cost_quote: float


@dataclass(frozen=True, slots=True)
class BacktestReport:
    """Complete outcome of one backtest run."""

    strategy_name: str
    capital: float
    equity_curve: np.ndarray
    returns: np.ndarray
    metrics: PerformanceMetrics
    fills: list[FillEvent] = field(default_factory=list)
    n_orders_generated: int = 0
    n_orders_executed: int = 0
    n_orders_skipped_min: int = 0
    n_orders_skipped_lot: int = 0
    total_cost_quote: float = 0.0
    total_cost_bps: float = 0.0
    total_funding_quote: float = 0.0
    n_funding_events_applied: int = 0

    @property
    def execution_rate(self) -> float:
        if self.n_orders_generated == 0:
            return 0.0
        return self.n_orders_executed / self.n_orders_generated
