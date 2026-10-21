"""
test_config.py

BacktestConfig construction, capital scenario loaders, periods_per_year.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

import pytest

from locus.backtest.config import (
    load_backtest_config,
    load_capital_scenario,
)


def test_load_capital_1() -> None:
    cfg = load_capital_scenario("1")
    assert cfg["capital"] == 1.0
    assert cfg["min_order_notional"] == 5.0


def test_load_capital_1m() -> None:
    cfg = load_capital_scenario("1m")
    assert cfg["capital"] == 1_000_000.0


def test_load_capital_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_capital_scenario("does_not_exist")


def test_periods_per_year() -> None:
    cfg = load_backtest_config(strategy_name="test", capital_name="10k", rebalance_freq_bars=24)
    assert cfg.periods_per_year == 365


def test_periods_per_year_weekly() -> None:
    cfg = load_backtest_config(strategy_name="test", capital_name="10k", rebalance_freq_bars=168)
    assert cfg.periods_per_year == 52


def test_lot_size_lookup() -> None:
    cfg = load_backtest_config(strategy_name="test", capital_name="10k")
    assert cfg.lot_size("BTC") == 0.00001
    assert cfg.lot_size("UNKNOWN") == cfg.default_lot_size
