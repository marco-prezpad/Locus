"""
comparison.py

Helpers to run multiple baselines on the same dataset and produce a
comparison table.

Funding (B4): passed through to the VectorizedEvaluator so all
strategies are compared net of funding.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

import polars as pl

from locus.baselines.base import BaselineStrategy, EvaluationResult
from locus.baselines.evaluator import VectorizedEvaluator
from locus.evaluation.validation import stationary_bootstrap_sharpe_ci
from locus.infrastructure.logging import LoggerFactory


def run_baselines(
    df: pl.DataFrame,
    strategies: list[BaselineStrategy],
    evaluator: VectorizedEvaluator,
    rebalance_freq: int,
    funding: pl.DataFrame | None = None,
) -> dict[str, EvaluationResult]:
    """Run all strategies on the same data. Returns name → result."""
    log = LoggerFactory.get("baselines.comparison")
    results: dict[str, EvaluationResult] = {}

    for strategy in strategies:
        log.info("strategy_started", extra={"strategy": strategy.name})
        weights = strategy.generate_weights(df, rebalance_freq)
        result = evaluator.evaluate(strategy.name, weights, df, funding=funding)
        results[strategy.name] = result
        log.info(
            "strategy_completed",
            extra={
                "strategy": strategy.name,
                "n_periods": result.metrics.n_periods,
                "sharpe": round(result.metrics.sharpe, 3),
                "total_return": round(result.metrics.total_return, 4),
                "funding_cost": round(result.total_funding_cost, 4),
            },
        )

    return results


def comparison_table(results: dict[str, EvaluationResult]) -> pl.DataFrame:
    """Build a wide DataFrame with one row per strategy and the main
    metrics as columns, including bootstrap CIs for Sharpe, sorted by
    Sharpe descending.

    Returns an empty DataFrame with the expected schema if `results` is
    empty.
    """
    rows = []
    for name, r in results.items():
        m = r.metrics

        # Sharpe CI: stationary block bootstrap. Requires >= 10 finite
        # returns; otherwise NaN (no silent crash, no fake interval).
        try:
            ci_low, ci_high = stationary_bootstrap_sharpe_ci(
                r.returns,
                r.periods_per_year,
                block_size=7,
                n_boot=2000,
                alpha=0.05,
            )
        except ValueError:
            ci_low, ci_high = float("nan"), float("nan")

        rows.append(
            {
                "strategy": name,
                "n_periods": m.n_periods,
                "total_return": m.total_return,
                "cagr": m.cagr,
                "volatility": m.volatility,
                "sharpe": m.sharpe,
                "sharpe_ci_low": ci_low,
                "sharpe_ci_high": ci_high,
                "sortino": m.sortino,
                "max_drawdown": m.max_drawdown,
                "calmar": m.calmar,
                "hit_rate": m.hit_rate,
                "turnover": m.turnover,
                "funding_cost": r.total_funding_cost,
            }
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "strategy": pl.Utf8,
                "n_periods": pl.Int64,
                "total_return": pl.Float64,
                "cagr": pl.Float64,
                "volatility": pl.Float64,
                "sharpe": pl.Float64,
                "sharpe_ci_low": pl.Float64,
                "sharpe_ci_high": pl.Float64,
                "sortino": pl.Float64,
                "max_drawdown": pl.Float64,
                "calmar": pl.Float64,
                "hit_rate": pl.Float64,
                "turnover": pl.Float64,
                "funding_cost": pl.Float64,
            }
        )

    return pl.DataFrame(rows).sort("sharpe", descending=True)
