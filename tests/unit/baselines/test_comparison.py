"""
test_comparison.py

run_baselines and comparison_table: strategy orchestration, wide
DataFrame construction, sorting by Sharpe.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from locus.baselines import (
    BuyAndHoldBTC,
    EqualWeightUniverse,
    VectorizedEvaluator,
    comparison_table,
    run_baselines,
)
from locus.data.providers.base import BAR_SCHEMA


def _df(n: int = 100, assets: tuple[str, ...] = ("BTC", "ETH")) -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for i, asset in enumerate(assets):
        for t in range(n):
            # Different growth rates so strategies differ
            price = 100.0 * (1 + 0.001 * (i + 1)) ** t
            rows.append(
                {
                    "asset_id": asset,
                    "venue": "test",
                    "symbol": f"{asset}USDT",
                    "timeframe": "1h",
                    "timestamp": base + timedelta(hours=t),
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1.0,
                }
            )
    return pl.DataFrame(rows, schema=BAR_SCHEMA)


def test_run_baselines_returns_dict_with_names() -> None:
    strategies = [BuyAndHoldBTC(), EqualWeightUniverse()]
    evaluator = VectorizedEvaluator(holding_period_bars=24, periods_per_year=365)
    results = run_baselines(_df(), strategies, evaluator, rebalance_freq=24)
    assert set(results.keys()) == {"buy_and_hold_btc", "equal_weight"}
    for r in results.values():
        assert r.metrics.n_periods > 0


def test_run_baselines_empty_list_returns_empty() -> None:
    evaluator = VectorizedEvaluator(holding_period_bars=24, periods_per_year=365)
    results = run_baselines(_df(), [], evaluator, rebalance_freq=24)
    assert results == {}


def test_comparison_table_columns() -> None:
    strategies = [BuyAndHoldBTC(), EqualWeightUniverse()]
    evaluator = VectorizedEvaluator(holding_period_bars=24, periods_per_year=365)
    results = run_baselines(_df(), strategies, evaluator, rebalance_freq=24)
    table = comparison_table(results)
    expected = {
        "strategy",
        "n_periods",
        "total_return",
        "cagr",
        "volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "hit_rate",
        "turnover",
        "funding_cost",
    }
    assert set(table.columns) == expected


def test_comparison_table_empty() -> None:
    table = comparison_table({})
    assert table.height == 0


def test_comparison_table_sorted_by_sharpe_descending() -> None:
    strategies = [BuyAndHoldBTC(), EqualWeightUniverse()]
    evaluator = VectorizedEvaluator(holding_period_bars=24, periods_per_year=365)
    results = run_baselines(_df(), strategies, evaluator, rebalance_freq=24)
    table = comparison_table(results)
    if table.height >= 2:
        sharpes = table["sharpe"].to_list()
        assert sharpes == sorted(sharpes, reverse=True)


def test_comparison_table_one_row_per_strategy() -> None:
    strategies = [BuyAndHoldBTC(), EqualWeightUniverse()]
    evaluator = VectorizedEvaluator(holding_period_bars=24, periods_per_year=365)
    results = run_baselines(_df(), strategies, evaluator, rebalance_freq=24)
    table = comparison_table(results)
    assert table.height == 2
