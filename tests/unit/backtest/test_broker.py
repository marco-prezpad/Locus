"""
test_broker.py

Broker: min-notional skip, lot rounding, cost application.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from locus.backtest.accounting import PortfolioAccounting
from locus.backtest.broker import SimulatedBroker
from locus.backtest.config import BacktestConfig
from locus.backtest.costs import (
    FeeModel,
    MarketImpactModel,
    SlippageModel,
    SpreadModel,
    TransactionCostModel,
)


def _config(capital: float, min_notional: float = 5.0) -> BacktestConfig:
    return BacktestConfig(
        strategy_name="test",
        capital=capital,
        currency="USDT",
        cost_model=TransactionCostModel(
            fee=FeeModel(10.0),
            spread=SpreadModel(5.0),
            slippage=SlippageModel(5.0),
            impact=MarketImpactModel(1.0),
        ),
        min_order_notional=min_notional,
        default_lot_size=1e-8,
        lot_sizes={"BTC": 0.00001, "ETH": 0.0001},
    )


def _setup(capital: float) -> tuple[SimulatedBroker, PortfolioAccounting]:
    config = _config(capital)
    return SimulatedBroker(config), PortfolioAccounting(cash=capital)


def _exec(
    broker: SimulatedBroker,
    accounting: PortfolioAccounting,
    weights: dict[str, float],
    equity: float,
) -> list:
    return broker.execute_rebalance(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        current_equity=equity,
        target_weights=weights,
        open_prices={"BTC": 30000.0, "ETH": 2000.0},
        sigma_daily={"BTC": 0.03, "ETH": 0.04},
        daily_volume_quote={"BTC": 1e9, "ETH": 1e8},
        accounting=accounting,
    )


def test_small_order_below_min_is_skipped() -> None:
    broker, acc = _setup(10.0)
    fills = _exec(broker, acc, {"BTC": 0.5}, equity=10.0)
    # 0.5 * $10 = $5 notional, exactly at min; lot rounding may drop it
    # to 0.0001 BTC = $3 → skip
    assert broker.n_skipped_min >= 1 or len(fills) == 0


def test_very_small_order_is_skipped() -> None:
    broker, acc = _setup(10.0)
    # 0.3 * $10 = $3 notional < $5 min
    fills = _exec(broker, acc, {"BTC": 0.3}, equity=10.0)
    assert fills == []
    assert broker.n_skipped_min == 1


def test_large_order_executes() -> None:
    broker, acc = _setup(10000.0)
    fills = _exec(broker, acc, {"BTC": 1.0}, equity=10000.0)
    assert len(fills) == 1
    assert fills[0].side == "buy"
    assert fills[0].asset_id == "BTC"


def test_lot_size_rounding() -> None:
    broker, acc = _setup(10000.0)
    _exec(broker, acc, {"BTC": 1.0}, equity=10000.0)
    # Target qty = 10000 / 30000 = 0.3333 BTC, rounded down to 0.00001
    assert acc.get_position("BTC") == pytest.approx(0.33333, abs=1e-5)


def test_cost_embedded_in_fill_price() -> None:
    broker, acc = _setup(10000.0)
    fills = _exec(broker, acc, {"BTC": 1.0}, equity=10000.0)
    # Buy: fill_price > open_price (cost added)
    assert fills[0].fill_price > 30000.0
    assert fills[0].cost_bps > 0
    assert fills[0].cost_quote > 0


def test_sell_at_lower_price_with_cost() -> None:
    broker, acc = _setup(10000.0)
    # First buy
    _exec(broker, acc, {"BTC": 1.0}, equity=10000.0)
    # Then sell to 0
    fills = _exec(broker, acc, {"BTC": 0.0}, equity=10000.0)
    assert len(fills) == 1
    assert fills[0].side == "sell"
    assert fills[0].fill_price < 30000.0


def test_no_change_no_fill() -> None:
    broker, acc = _setup(10000.0)
    _exec(broker, acc, {"BTC": 1.0}, equity=10000.0)
    fills = _exec(broker, acc, {"BTC": 1.0}, equity=10000.0)
    assert fills == []
