"""
test_evaluator.py

One-bar-delay, holding-period returns, cash handling, turnover.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.baselines import VectorizedEvaluator
from locus.data.providers.base import BAR_SCHEMA


def _ohlcv(prices: list[float], asset: str = "BTC") -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "asset_id": asset,
                "venue": "test",
                "symbol": f"{asset}USDT",
                "timeframe": "1h",
                "timestamp": base + timedelta(hours=i),
                "open": p,
                "high": p,
                "low": p,
                "close": p,
                "volume": 1.0,
            }
            for i, p in enumerate(prices)
        ],
        schema=BAR_SCHEMA,
    )


def _weights(timestamps: list[datetime], weights: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["BTC"] * len(timestamps),
            "weight": weights,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def test_one_bar_delay_return() -> None:
    """Price T=0:100, T=1:110, T=2:121, T=3:133.1.
    Weight at T=0 → execute at open(1)=110, exit at open(1+h) with h=1 → open(2)=121.
    Return = 121/110 - 1 = 0.1.
    """
    ohlcv = _ohlcv([100.0, 110.0, 121.0, 133.1])
    ts = ohlcv["timestamp"].to_list()
    weights = _weights([ts[0]], [1.0])
    evaluator = VectorizedEvaluator(holding_period_bars=1, periods_per_year=1)
    result = evaluator.evaluate("test", weights, ohlcv)
    assert result.metrics.n_periods == 1
    assert result.returns[0] == pytest.approx(0.1, abs=1e-9)


def test_holding_period_two_bars() -> None:
    """Weight at T=0 → open(1)=110 → open(3)=133.1 → return 21%."""
    ohlcv = _ohlcv([100.0, 110.0, 121.0, 133.1, 146.41])
    ts = ohlcv["timestamp"].to_list()
    weights = _weights([ts[0]], [1.0])
    evaluator = VectorizedEvaluator(holding_period_bars=2, periods_per_year=1)
    result = evaluator.evaluate("test", weights, ohlcv)
    assert result.returns[0] == pytest.approx(133.1 / 110.0 - 1.0, abs=1e-9)


def test_cash_reduces_exposure() -> None:
    """Weight 0.5 on an asset: only half the return is applied."""
    ohlcv = _ohlcv([100.0, 110.0, 121.0])
    ts = ohlcv["timestamp"].to_list()
    weights = _weights([ts[0]], [0.5])
    evaluator = VectorizedEvaluator(holding_period_bars=1, periods_per_year=1)
    result = evaluator.evaluate("test", weights, ohlcv)
    assert result.returns[0] == pytest.approx(0.5 * 0.1, abs=1e-9)


def test_empty_weights_returns_empty() -> None:
    ohlcv = _ohlcv([100.0, 110.0])
    empty_weights = pl.DataFrame(
        schema={"timestamp": pl.Datetime("us", "UTC"), "asset_id": pl.Utf8, "weight": pl.Float64}
    )
    result = VectorizedEvaluator(1).evaluate("test", empty_weights, ohlcv)
    assert result.metrics.n_periods == 0


def test_rejects_zero_holding_period() -> None:
    with pytest.raises(ValueError):
        VectorizedEvaluator(holding_period_bars=0)
