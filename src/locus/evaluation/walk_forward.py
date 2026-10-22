"""
walk_forward.py

WalkForwardBaselineEvaluator: runs a baseline strategy across K
contiguous, non-overlapping time windows and concatenates the per-window
returns into a single series.

Why windows for naive baselines: the strategy has no trainable
parameters, so there is no "training" to walk forward. What we want to
verify is **stability** — does the strategy's Sharpe hold up across
different market regimes? — and **statistical power** — does the
concatenated series have enough observations to test significance?

The per-window Sharpe is reported as a stability check; the concatenated
returns feed DSR / CI / MinBTL.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from locus.backtest.broker import SimulatedBroker
from locus.backtest.config import BacktestConfig
from locus.backtest.engine import BacktestEngine
from locus.evaluation.metrics import PerformanceMetrics, compute_metrics
from locus.infrastructure.logging import LoggerFactory

if TYPE_CHECKING:
    from locus.baselines.base import BaselineStrategy


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Outcome of one window of the walk-forward."""

    fold: int
    start: datetime
    end: datetime
    n_returns: int
    metrics: PerformanceMetrics


@dataclass(frozen=True, slots=True)
class WalkForwardStrategyResult:
    """Full walk-forward outcome for one strategy."""

    strategy_name: str
    per_fold: list[FoldResult]
    returns: np.ndarray  # concatenated
    metrics: PerformanceMetrics  # on concatenated series
    periods_per_year: int


class WalkForwardBaselineEvaluator:
    """Run a baseline over K contiguous windows and aggregate."""

    def __init__(
        self,
        config_template: BacktestConfig,
        n_folds: int = 4,
    ) -> None:
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        self._template = config_template
        self._n_folds = n_folds
        self._log = LoggerFactory.get("evaluation.walk_forward")

    @property
    def n_folds(self) -> int:
        return self._n_folds

    def run(
        self,
        strategy: BaselineStrategy,
        ohlcv: pl.DataFrame,
        rebalance_freq_bars: int,
    ) -> WalkForwardStrategyResult:
        """Split the OHLCV timeline into n_folds contiguous, non-overlapping
        windows; run the strategy in each; concatenate per-period returns."""
        timestamps = sorted(ohlcv["timestamp"].unique().to_list())
        n = len(timestamps)
        if n < self._n_folds * 2:
            raise ValueError(f"not enough timestamps ({n}) for {self._n_folds} folds")

        boundaries: list[tuple[datetime, datetime]] = []
        step = n // self._n_folds
        for k in range(self._n_folds):
            start_idx = k * step
            end_idx = (k + 1) * step - 1 if k < self._n_folds - 1 else n - 1
            boundaries.append((timestamps[start_idx], timestamps[end_idx]))

        per_fold: list[FoldResult] = []
        all_returns: list[np.ndarray] = []

        for k, (start_ts, end_ts) in enumerate(boundaries):
            window = ohlcv.filter(
                (pl.col("timestamp") >= start_ts) & (pl.col("timestamp") <= end_ts)
            )
            weights = strategy.generate_weights(window, rebalance_freq_bars)

            cfg = BacktestConfig(
                strategy_name=f"{strategy.name}_fold{k}",
                capital=self._template.capital,
                currency=self._template.currency,
                cost_model=self._template.cost_model,
                min_order_notional=self._template.min_order_notional,
                default_lot_size=self._template.default_lot_size,
                lot_sizes=self._template.lot_sizes,
                bars_per_day=self._template.bars_per_day,
                rebalance_freq_bars=self._template.rebalance_freq_bars,
                risk_free_rate_annual=self._template.risk_free_rate_annual,
            )
            engine = BacktestEngine(cfg, SimulatedBroker(cfg))
            report = engine.run(weights, window)

            per_fold.append(
                FoldResult(
                    fold=k,
                    start=start_ts,
                    end=end_ts,
                    n_returns=int(report.metrics.n_periods),
                    metrics=report.metrics,
                )
            )
            if len(report.returns) > 0:
                all_returns.append(report.returns)

        concatenated = np.concatenate(all_returns) if all_returns else np.array([])
        aggregate = compute_metrics(
            concatenated,
            self._template.periods_per_year,
            self._template.risk_free_rate_annual,
        )

        self._log.info(
            "walk_forward_completed",
            extra={
                "strategy": strategy.name,
                "n_folds": self._n_folds,
                "n_returns": len(concatenated),
                "sharpe_aggregate": round(float(aggregate.sharpe), 3),
            },
        )

        return WalkForwardStrategyResult(
            strategy_name=strategy.name,
            per_fold=per_fold,
            returns=concatenated,
            metrics=aggregate,
            periods_per_year=self._template.periods_per_year,
        )
