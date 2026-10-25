"""
base.py

The BaselineStrategy Protocol and the EvaluationResult contract.

A strategy produces a weights DataFrame with columns
(timestamp, asset_id, weight) at rebalance timestamps. Weights sum to
<= 1 at each timestamp (the difference is cash, which earns 0).

The evaluator is strategy-agnostic: it takes weights + OHLCV (+ optional
funding) and produces an equity curve. This separation lets F6
(event-driven backtester) and F9 (ML models) reuse the same weights
interface.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from locus.evaluation.metrics import PerformanceMetrics


class BaselineStrategy(Protocol):
    """Generate portfolio weights at rebalance timestamps."""

    @property
    def name(self) -> str: ...

    def generate_weights(self, df: pl.DataFrame, rebalance_freq: int) -> pl.DataFrame:
        """Return DataFrame[timestamp, asset_id, weight] with one row per
        (rebalance_timestamp, asset).

        - `df` has BAR_SCHEMA plus features (e.g., return_20 for momentum).
        - `rebalance_freq` is the number of bars between rebalances.
        - Rebalance timestamps start at the first timestamp with all
          required features available and advance by `rebalance_freq` bars.
        - Weights sum to <= 1 at each timestamp (cash is the rest).
        """
        ...


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Complete evaluation of one strategy."""

    strategy_name: str
    equity_curve: np.ndarray  # shape (n+1,), starts at 1.0
    returns: np.ndarray  # shape (n,), per-rebalance NET returns
    weights_used: pl.DataFrame  # (timestamp, asset_id, weight, asset_return)
    metrics: PerformanceMetrics
    periods_per_year: int
    total_funding_cost: float = 0.0  # signed: positive means funding paid
    n_funding_events: int = 0
