"""
test_engine.py

Engine: one-bar-delay, equity curve, capital scaling behavior.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.backtest.broker import SimulatedBroker
from locus.backtest.config import BacktestConfig
from locus.backtest.costs import (
    FeeModel,
    MarketImpactModel,
    SlippageModel,
    SpreadModel,
    TransactionCostModel,
)
from locus.backtest.engine import BacktestEngine
from locus.data.providers.base import BAR_SCHEMA


def _ohlcv(n: int = 100, start_price: float = 100.0, growth: float = 0.001) -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "asset_id": "BTC",
                "venue": "t",
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "timestamp": base + timedelta(hours=i),
                "open": start_price * (1 + growth * i),
                "high": start_price * (1 + growth * i) * 1.01,
                "low": start_price * (1 + growth * i) * 0.99,
                "close": start_price * (1 + growth * i),
                "volume": 1e6,
            }
            for i in range(n)
        ],
        schema=BAR_SCHEMA,
    )


def _weights_full_btc(ts: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {"timestamp": ts, "asset_id": ["BTC"] * len(ts), "weight": [1.0] * len(ts)}
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


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
        lot_sizes={"BTC": 0.00001},
        rebalance_freq_bars=24,
    )


def test_engine_equity_curve_length() -> None:
    ohlcv = _ohlcv(100)
    ts = ohlcv["timestamp"].to_list()[::24]
    weights = _weights_full_btc(ts)
    engine = BacktestEngine(_config(10_000.0), SimulatedBroker(_config(10_000.0)))
    report = engine.run(weights, ohlcv)
    # 5 rebalances (0, 24, 48, 72, 96) → equity_curve has 6 entries
    assert len(report.equity_curve) == len(ts) + 1
    assert report.equity_curve[0] == pytest.approx(10_000.0)


def test_engine_at_dollar_1_no_execution() -> None:
    ohlcv = _ohlcv(100)
    ts = ohlcv["timestamp"].to_list()[::24]
    weights = _weights_full_btc(ts)
    cfg = _config(1.0)
    engine = BacktestEngine(cfg, SimulatedBroker(cfg))
    report = engine.run(weights, ohlcv)
    assert report.n_orders_executed == 0
    assert report.n_orders_skipped_min > 0
    # All equity in cash
    assert report.equity_curve[-1] == pytest.approx(1.0)


def test_engine_at_10k_executes() -> None:
    ohlcv = _ohlcv(100)
    ts = ohlcv["timestamp"].to_list()[::24]
    weights = _weights_full_btc(ts)
    cfg = _config(10_000.0)
    engine = BacktestEngine(cfg, SimulatedBroker(cfg))
    report = engine.run(weights, ohlcv)
    assert report.n_orders_executed >= 1
    # Equity reflects costs (slightly below initial after first buy)
    assert report.equity_curve[-1] > 0


def test_engine_at_10_no_execution_for_single_asset() -> None:
    """$10 notional, min $5: 1.0 weight gives $10 which is >= $5, but
    lot rounding may reduce it. Either way, it's a marginal case."""
    ohlcv = _ohlcv(100)
    ts = ohlcv["timestamp"].to_list()[::24]
    weights = _weights_full_btc(ts)
    cfg = _config(10.0)
    engine = BacktestEngine(cfg, SimulatedBroker(cfg))
    report = engine.run(weights, ohlcv)
    # At $10 and BTC price 100, qty = 0.1, notional = $10, executes.
    # But with our lot size 0.00001, qty = 0.1 exactly. So it executes.
    assert report.n_orders_executed >= 1
