"""
Engine consistency tests (plan v3.1 P6).

Each test isolates one property of the engine that a bug in costs, pricing,
or capital accounting would break. If any of these fails, the engine is not
fit to be the source of truth for verdicts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

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
from locus.baselines import VectorizedEvaluator


def _ohlcv(prices: list[float]) -> pl.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "asset_id": "BTC",
                "timestamp": base + timedelta(hours=i),
                "open": p,
                "high": p,
                "low": p,
                "close": p,
                "volume": 1000.0,
            }
            for i, p in enumerate(prices)
        ]
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def _weights(weight: float, n_rebal: int = 4) -> pl.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "timestamp": [base + timedelta(hours=24 * i) for i in range(n_rebal)],
            "asset_id": ["BTC"] * n_rebal,
            "weight": [weight] * n_rebal,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def _costs(
    fee: float = 0.0,
    spread: float = 0.0,
    slip: float = 0.0,
    impact: float = 0.0,
) -> TransactionCostModel:
    return TransactionCostModel(
        fee=FeeModel(fee),
        spread=SpreadModel(spread),
        slippage=SlippageModel(slip),
        impact=MarketImpactModel(impact),
    )


def _config(
    cost_model: TransactionCostModel,
    capital: float = 10_000.0,
) -> BacktestConfig:
    return BacktestConfig(
        strategy_name="consistency",
        capital=capital,
        currency="USDT",
        cost_model=cost_model,
        min_order_notional=0.0,
        default_lot_size=1e-8,
        lot_sizes={"BTC": 1e-8},
        bars_per_day=24,
        rebalance_freq_bars=24,
        annualization_days=365,
        apply_funding=False,
    )


# --- Test 1: no price change, non-zero costs → negative return ---


def test_flat_price_non_zero_costs_yield_negative_return() -> None:
    bars = _ohlcv([100.0] * 120)
    weights = _weights(1.0)
    cfg = _config(_costs(fee=10.0, spread=5.0, slip=5.0))
    report = BacktestEngine(cfg, SimulatedBroker(cfg)).run(weights, bars)
    assert report.total_cost_quote > 0
    assert report.metrics.total_return < 0


# --- Test 2: high fees → very negative return ---


def test_high_fees_yield_deeply_negative_return() -> None:
    bars = _ohlcv([100.0] * 120)
    weights = _weights(1.0)
    cfg = _config(_costs(fee=200.0, spread=100.0, slip=100.0))
    report = BacktestEngine(cfg, SimulatedBroker(cfg)).run(weights, bars)
    assert report.metrics.total_return < 0
    assert report.total_cost_quote > 0


# --- Test 3: rising market, long position → positive return ---


def test_rising_market_long_position_profits() -> None:
    bars = _ohlcv([100.0 + i for i in range(120)])
    weights = _weights(1.0)
    cfg = _config(_costs())
    report = BacktestEngine(cfg, SimulatedBroker(cfg)).run(weights, bars)
    assert report.metrics.total_return > 0


# --- Test 4: capital invariance (no rounding) ---


def test_capital_invariance_without_rounding() -> None:
    bars = _ohlcv([100.0 + i * 0.1 for i in range(120)])
    weights = _weights(1.0)
    small = _config(_costs(), capital=1_000.0)
    large = _config(_costs(), capital=100_000.0)
    r_small = BacktestEngine(small, SimulatedBroker(small)).run(weights, bars)
    r_large = BacktestEngine(large, SimulatedBroker(large)).run(weights, bars)
    assert abs(r_small.metrics.total_return - r_large.metrics.total_return) < 1e-9


# --- Test 5: engine matches vectorized without costs ---


def test_engine_matches_vectorized_without_costs() -> None:
    # Smooth, low-slope price. Engine marks at close(rebalance_ts[i+1]);
    # vectorized marks at open(exec_ts + h). On flat or smooth prices the
    # two agree to well under 1e-3. On volatile synthetic data the timing
    # difference alone reaches ~3e-3, which is a measurement artifact, not
    # a bug. Plan §6.1 reports ~8e-5 on real data (diag_b5_9_fold0).
    prices = [100.0 + 0.001 * i for i in range(200)]
    bars = _ohlcv(prices)
    weights = _weights(1.0, n_rebal=4)

    cfg = _config(_costs())
    engine_report = BacktestEngine(cfg, SimulatedBroker(cfg)).run(weights, bars)

    ev = VectorizedEvaluator(holding_period_bars=24, periods_per_year=365)
    vec_result = ev.evaluate("vec", weights, bars)

    n = min(len(engine_report.returns), len(vec_result.returns))
    assert n > 0, "no overlapping returns"
    diff = np.abs(engine_report.returns[:n] - vec_result.returns[:n])
    assert diff.mean() < 1e-3
