"""
test_funding.py

Contract tests for funding cash flows in the BacktestEngine (B3).

Sign convention: positive funding rate => longs pay shorts.
    cash_flow = -qty * price * funding_rate

Funding must NOT affect fill prices.

Design: single rebalance (open + hold) so the test isolates the funding
cash flow from rebalance-induced quantity changes. With funding reducing
equity, a second rebalance to weight=1.0 would trigger a sell that makes
the expected total funding a moving target.

Author: Marco Pérez Padilla
Date:   25-10-2026
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
from locus.backtest.events import BacktestReport


def _ohlcv_one_asset(
    n_bars: int = 4,
    start: datetime = datetime(2026, 9, 30, tzinfo=UTC),
    step_hours: int = 8,
    price: float = 100.0,
) -> pl.DataFrame:
    rows = []
    for i in range(n_bars):
        ts = start + timedelta(hours=step_hours * i)
        rows.append(
            {
                "asset_id": "BTC",
                "timestamp": ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1_000.0,
            }
        )
    return pl.DataFrame(rows)


def _funding_one_asset(
    rate: float,
    start: datetime = datetime(2026, 9, 30, tzinfo=UTC),
    step_hours: int = 8,
    n_bars: int = 4,
) -> pl.DataFrame:
    rows = []
    for i in range(n_bars):
        ts = start + timedelta(hours=step_hours * i)
        rows.append(
            {
                "asset_id": "BTC",
                "venue": "binance_futures",
                "symbol": "BTCUSDT",
                "timestamp": ts,
                "funding_rate": rate,
            }
        )
    return pl.DataFrame(rows)


def _weights_one_asset(n_rebalances: int = 1, weight: float = 1.0) -> pl.DataFrame:
    start = datetime(2026, 9, 30, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "asset_id": "BTC",
                "timestamp": start + timedelta(hours=8 * i),
                "weight": weight,
            }
            for i in range(n_rebalances)
        ]
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def _config(capital: float = 1_000.0, apply_funding: bool = True) -> BacktestConfig:
    return BacktestConfig(
        strategy_name="test_funding",
        capital=capital,
        currency="USDT",
        cost_model=TransactionCostModel(
            fee=FeeModel(0.0),
            spread=SpreadModel(0.0),
            slippage=SlippageModel(0.0),
            impact=MarketImpactModel(0.0),
        ),
        min_order_notional=0.0,
        default_lot_size=1e-8,
        lot_sizes={"BTC": 1e-8},
        rebalance_freq_bars=8,
        apply_funding=apply_funding,
    )


def _run(
    funding_rate: float,
    weight: float,
    capital: float = 1_000.0,
) -> BacktestReport:
    ohlcv = _ohlcv_one_asset(n_bars=4, price=100.0)
    funding = _funding_one_asset(rate=funding_rate, n_bars=4) if funding_rate != 0.0 else None
    weights = _weights_one_asset(n_rebalances=1, weight=weight)
    cfg = _config(capital=capital)
    engine = BacktestEngine(cfg, SimulatedBroker(cfg))
    return engine.run(weights, ohlcv, funding=funding)


def test_funding_cash_flow_sign_long_pays() -> None:
    """Long + positive funding => total_funding_quote < 0."""
    report = _run(funding_rate=0.001, weight=1.0)
    # Single rebalance at 00:00, fill at 08:00 (10 BTC @ 100).
    # Funding at 00:00 skipped (pre-fill, qty 0); funding at 08:00, 16:00,
    # 00:00+1 applies to qty 10, price 100, rate 0.001 → -1.0 each.
    assert report.total_funding_quote == pytest.approx(-3.0, abs=1e-9)
    assert report.n_funding_events_applied == 3
    assert report.equity_curve[-1] < report.capital


def test_funding_cash_flow_sign_short_receives() -> None:
    """Short + positive funding => total_funding_quote > 0."""
    report = _run(funding_rate=0.001, weight=-1.0)
    assert report.total_funding_quote == pytest.approx(+3.0, abs=1e-9)
    assert report.n_funding_events_applied == 3


def test_funding_cash_flow_does_not_affect_fill_price() -> None:
    """Funding is a separate cash flow: fill prices must be identical
    with and without funding events."""
    report_no_funding = _run(funding_rate=0.0, weight=1.0)
    report_with_funding = _run(funding_rate=0.001, weight=1.0)
    assert len(report_no_funding.fills) == len(report_with_funding.fills)
    for f0, f1 in zip(report_no_funding.fills, report_with_funding.fills, strict=True):
        assert f0.fill_price == pytest.approx(f1.fill_price)
        assert f0.quantity == pytest.approx(f1.quantity)


def test_funding_not_applied_when_none() -> None:
    """If funding=None, no funding events are applied."""
    ohlcv = _ohlcv_one_asset(n_bars=4, price=100.0)
    weights = _weights_one_asset(n_rebalances=1, weight=1.0)
    cfg = _config()
    engine = BacktestEngine(cfg, SimulatedBroker(cfg))
    report = engine.run(weights, ohlcv, funding=None)
    assert report.total_funding_quote == 0.0
    assert report.n_funding_events_applied == 0


def test_funding_not_applied_when_config_flag_is_false() -> None:
    """If config.apply_funding=False, funding DataFrame is ignored."""
    ohlcv = _ohlcv_one_asset(n_bars=4, price=100.0)
    funding = _funding_one_asset(rate=0.001, n_bars=4)
    weights = _weights_one_asset(n_rebalances=1, weight=1.0)
    cfg = _config(apply_funding=False)
    engine = BacktestEngine(cfg, SimulatedBroker(cfg))
    report = engine.run(weights, ohlcv, funding=funding)
    assert report.total_funding_quote == 0.0
    assert report.n_funding_events_applied == 0
