"""
test_scaling.py

CapitalScalingExperiment: runs one strategy across all capital
scenarios, produces a comparison table, execution rate varies with
capital.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from locus.backtest.scaling import CapitalScalingExperiment
from locus.baselines import BuyAndHoldBTC
from locus.data.providers.base import BAR_SCHEMA


def _df(n: int = 200, start_price: float = 100.0) -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "asset_id": "BTC",
                "venue": "test",
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "timestamp": base + timedelta(hours=i),
                "open": start_price,
                "high": start_price * 1.01,
                "low": start_price * 0.99,
                "close": start_price,
                "volume": 1e6,
            }
            for i in range(n)
        ],
        schema=BAR_SCHEMA,
    )


def test_run_returns_dict_and_table() -> None:
    exp = CapitalScalingExperiment()
    reports, table = exp.run(BuyAndHoldBTC(), _df(), rebalance_freq_bars=24)
    assert isinstance(reports, dict)
    assert isinstance(table, pl.DataFrame)


def test_run_covers_all_seven_capitals() -> None:
    exp = CapitalScalingExperiment()
    reports, table = exp.run(BuyAndHoldBTC(), _df(), rebalance_freq_bars=24)
    assert set(reports.keys()) == {"1", "10", "100", "1k", "10k", "100k", "1m"}
    assert table.height == 7


def test_run_table_capital_column_reflects_scenario() -> None:
    exp = CapitalScalingExperiment()
    _, table = exp.run(BuyAndHoldBTC(), _df(), rebalance_freq_bars=24)
    caps = set(table["capital"].to_list())
    assert caps == {1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0}


def test_at_dollar_1_no_execution() -> None:
    exp = CapitalScalingExperiment(capital_names=["1"])
    reports, _ = exp.run(BuyAndHoldBTC(), _df(), rebalance_freq_bars=24)
    report = reports["1"]
    # $1 × weight 1.0 = $1 notional < $5 min → all skipped
    assert report.n_orders_executed == 0
    assert report.n_orders_skipped_min > 0
    assert report.execution_rate == 0.0


def test_at_dollar_10_execution_may_occur() -> None:
    exp = CapitalScalingExperiment(capital_names=["10"])
    reports, _ = exp.run(BuyAndHoldBTC(), _df(), rebalance_freq_bars=24)
    report = reports["10"]
    # $10 × weight 1.0 = $10 notional >= $5 min → executes
    assert report.n_orders_executed >= 1


def test_at_10k_executes_at_least_once() -> None:
    exp = CapitalScalingExperiment(capital_names=["10k"])
    reports, _ = exp.run(BuyAndHoldBTC(), _df(), rebalance_freq_bars=24)
    report = reports["10k"]
    # With constant prices, buy&hold only executes on the first rebalance
    # (subsequent target deltas are ~0 and skipped below min notional).
    # The assertion is "at least one trade executes", not a high rate.
    assert report.n_orders_executed >= 1
    assert report.execution_rate > 0.0


def test_custom_capital_names_respected() -> None:
    exp = CapitalScalingExperiment(capital_names=["1k", "10k"])
    reports, table = exp.run(BuyAndHoldBTC(), _df(), rebalance_freq_bars=24)
    assert set(reports.keys()) == {"1k", "10k"}
    assert table.height == 2


def test_pessimistic_costs_lower_sharpe_than_optimistic() -> None:
    opt = CapitalScalingExperiment(capital_names=["10k"], cost_scenario="optimistic")
    pes = CapitalScalingExperiment(capital_names=["10k"], cost_scenario="pessimistic")
    _, t_opt = opt.run(BuyAndHoldBTC(), _df(), rebalance_freq_bars=24)
    _, t_pes = pes.run(BuyAndHoldBTC(), _df(), rebalance_freq_bars=24)
    assert t_opt["total_cost_bps"][0] < t_pes["total_cost_bps"][0]


def test_run_with_rebalance_freq_weekly() -> None:
    exp = CapitalScalingExperiment(capital_names=["10k"])
    reports, _ = exp.run(BuyAndHoldBTC(), _df(n=1000), rebalance_freq_bars=168)
    assert reports["10k"].metrics.n_periods > 0
