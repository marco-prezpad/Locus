"""
evaluator.py

VectorizedEvaluator: applies weights to OHLCV with one-bar-delay and
computes holding-period returns.

One-bar-delay: weights computed from features at close of T execute at
open(T+1). Holding: until open(T+1+h), where h = holding_period_bars.
Return: open_i(T+1+h) / open_i(T+1) - 1 per asset; portfolio return =
sum_i w_i * asset_return_i. Cash contributes 0.

Funding (B4): when a funding DataFrame is provided, each holding period
pays/receives the sum of funding rates whose timestamp falls in
(exec_ts, exit_ts], weighted by the period's weight. Sign convention
matches B3: positive funding rate => longs pay shorts.

    net_return = gross_return - sum_i w_i * sum(funding_rate_i)

A positive weight (long) with positive funding rate reduces the return;
a negative weight (short) with positive funding rate increases it.
Funding does NOT affect asset prices or execution.

Requires all assets to have bars at all timestamps (aligned). Gappy data
is dropped from the per-row join; a missing execution/exit price for an
asset with weight > 0 silently reduces portfolio exposure.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

import numpy as np
import polars as pl

from locus.baselines.base import EvaluationResult
from locus.evaluation.metrics import compute_metrics
from locus.infrastructure.logging import LoggerFactory


class VectorizedEvaluator:
    """Evaluate a weights DataFrame against OHLCV (and optional funding)."""

    def __init__(
        self,
        holding_period_bars: int,
        periods_per_year: int = 365,
        risk_free_rate_annual: float = 0.0,
    ) -> None:
        if holding_period_bars <= 0:
            raise ValueError("holding_period_bars must be positive")
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        self._h = holding_period_bars
        self._ppy = periods_per_year
        self._rf = risk_free_rate_annual
        self._log = LoggerFactory.get("baselines.evaluator")

    def evaluate(
        self,
        strategy_name: str,
        weights_df: pl.DataFrame,
        ohlcv_df: pl.DataFrame,
        funding: pl.DataFrame | None = None,
    ) -> EvaluationResult:
        if weights_df.height == 0:
            return EvaluationResult(
                strategy_name=strategy_name,
                equity_curve=np.array([1.0]),
                returns=np.array([]),
                weights_used=pl.DataFrame(),
                metrics=compute_metrics(np.array([]), self._ppy, self._rf),
                periods_per_year=self._ppy,
            )

        ohlcv_sorted = ohlcv_df.sort(["asset_id", "timestamp"])
        ohlcv_ext = ohlcv_sorted.with_columns(
            [
                pl.col("open").shift(-1).over("asset_id").alias("exec_open"),
                pl.col("open").shift(-1 - self._h).over("asset_id").alias("exit_open"),
                pl.col("timestamp").shift(-1).over("asset_id").alias("exec_ts"),
                pl.col("timestamp").shift(-1 - self._h).over("asset_id").alias("exit_ts"),
            ]
        ).select(["asset_id", "timestamp", "exec_open", "exit_open", "exec_ts", "exit_ts"])

        joined = weights_df.join(ohlcv_ext, on=["asset_id", "timestamp"], how="inner")
        joined = (
            joined.with_columns(
                (pl.col("exit_open") / pl.col("exec_open") - 1.0).alias("asset_return")
            )
            .filter(pl.col("exec_open").is_not_null() & pl.col("exit_open").is_not_null())
            .sort(["timestamp", "asset_id"])
            .with_row_index("_row")
        )

        if joined.height == 0:
            self._log.warning(
                "empty_after_join",
                extra={"strategy": strategy_name, "weights_rows": weights_df.height},
            )
            return EvaluationResult(
                strategy_name=strategy_name,
                equity_curve=np.array([1.0]),
                returns=np.array([]),
                weights_used=joined,
                metrics=compute_metrics(np.array([]), self._ppy, self._rf),
                periods_per_year=self._ppy,
            )

        joined, n_funding_events = self._attach_funding(joined, funding)

        portfolio = (
            joined.group_by("timestamp")
            .agg(
                (pl.col("weight") * pl.col("asset_return")).sum().alias("gross_return"),
                (pl.col("weight") * pl.col("funding_rate_sum")).sum().alias("funding_cost"),
            )
            .sort("timestamp")
            .with_columns(
                (pl.col("gross_return") - pl.col("funding_cost")).alias("portfolio_return")
            )
        )

        returns = portfolio["portfolio_return"].to_numpy()
        total_funding_cost = float(portfolio["funding_cost"].sum())
        turnover = _compute_turnover(weights_df)
        metrics = compute_metrics(returns, self._ppy, self._rf, turnover)
        equity = np.concatenate([[1.0], np.cumprod(1.0 + returns)])

        return EvaluationResult(
            strategy_name=strategy_name,
            equity_curve=equity,
            returns=returns,
            weights_used=joined,
            metrics=metrics,
            periods_per_year=self._ppy,
            total_funding_cost=total_funding_cost,
            n_funding_events=n_funding_events,
        )

    @staticmethod
    def _attach_funding(
        joined: pl.DataFrame,
        funding: pl.DataFrame | None,
    ) -> tuple[pl.DataFrame, int]:
        """Attach `funding_rate_sum` per weight row (sum of events in
        (exec_ts, exit_ts]) and return the number of events matched."""
        if funding is None or funding.height == 0:
            return joined.with_columns(pl.lit(0.0).alias("funding_rate_sum")), 0

        f = (
            funding.select(["asset_id", "timestamp", "funding_rate"])
            .rename({"timestamp": "funding_ts"})
            .with_columns(pl.col("funding_ts").dt.truncate("1m").alias("funding_ts"))
        )

        intervals = joined.select(["_row", "asset_id", "exec_ts", "exit_ts"]).with_columns(
            [
                pl.col("exec_ts").dt.truncate("1m").alias("exec_ts"),
                pl.col("exit_ts").dt.truncate("1m").alias("exit_ts"),
            ]
        )

        matched = intervals.join(f, on="asset_id", how="inner").filter(
            (pl.col("funding_ts") > pl.col("exec_ts")) & (pl.col("funding_ts") <= pl.col("exit_ts"))
        )

        agg = matched.group_by("_row").agg(pl.col("funding_rate").sum().alias("funding_rate_sum"))

        out = joined.join(agg, on="_row", how="left").with_columns(
            pl.col("funding_rate_sum").fill_null(0.0)
        )
        return out, int(matched.height)


def _compute_turnover(weights_df: pl.DataFrame) -> float:
    """Average per-rebalance turnover: sum_i |w[r,i] - w[r-1,i]| / 2,
    averaged over rebalances. Assets missing at a timestamp are treated
    as weight 0."""
    if weights_df.height == 0:
        return 0.0

    pivot = weights_df.pivot(index="timestamp", on="asset_id", values="weight").fill_null(0.0)

    asset_cols = [c for c in pivot.columns if c != "timestamp"]
    if not asset_cols:
        return 0.0

    w = pivot.select(asset_cols).to_numpy()
    if w.shape[0] < 2:
        return 0.0

    dw = np.abs(np.diff(w, axis=0))
    per_period = dw.sum(axis=1) / 2.0
    return float(per_period.mean())
