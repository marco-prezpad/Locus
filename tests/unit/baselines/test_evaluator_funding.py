"""
test_evaluator_funding.py

Contract tests for funding in the vectorized evaluator (B4).

Sign convention (same as B3): positive funding rate => longs pay shorts.
    net_return = gross_return - weight * sum(funding_rate in holding period)

Author: Marco Pérez Padilla
Date:   25-10-2026
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


def _funding(
    timestamps: list[datetime],
    rates: list[float],
    asset: str = "BTC",
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "asset_id": [asset] * len(timestamps),
            "venue": ["binance_futures"] * len(timestamps),
            "symbol": [f"{asset}USDT"] * len(timestamps),
            "timestamp": timestamps,
            "funding_rate": rates,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def test_funding_zero_rate_leaves_returns_unchanged() -> None:
    ohlcv = _ohlcv([100.0, 110.0, 121.0, 133.1])
    ts = ohlcv["timestamp"].to_list()
    weights = _weights([ts[0]], [1.0])
    funding = _funding([ts[2]], [0.0])

    evaluator = VectorizedEvaluator(holding_period_bars=2, periods_per_year=1)
    result = evaluator.evaluate("test", weights, ohlcv, funding=funding)

    assert result.returns[0] == pytest.approx(133.1 / 110.0 - 1.0, abs=1e-9)
    assert result.total_funding_cost == pytest.approx(0.0)
    assert result.n_funding_events == 1


def test_funding_long_pays_positive_rate() -> None:
    """Long + positive funding => net return < gross return."""
    ohlcv = _ohlcv([100.0, 100.0, 100.0, 100.0, 100.0])
    ts = ohlcv["timestamp"].to_list()
    weights = _weights([ts[0]], [1.0])
    # holding_period_bars=2 → interval (ts[1], ts[3]]
    funding = _funding([ts[2]], [0.001])

    evaluator = VectorizedEvaluator(holding_period_bars=2, periods_per_year=1)
    result = evaluator.evaluate("test", weights, ohlcv, funding=funding)

    assert result.returns[0] == pytest.approx(-0.001, abs=1e-9)
    assert result.total_funding_cost == pytest.approx(0.001, abs=1e-12)


def test_funding_short_receives_positive_rate() -> None:
    """Short + positive funding => net return > gross return."""
    ohlcv = _ohlcv([100.0, 100.0, 100.0, 100.0, 100.0])
    ts = ohlcv["timestamp"].to_list()
    weights = _weights([ts[0]], [-1.0])
    funding = _funding([ts[2]], [0.001])

    evaluator = VectorizedEvaluator(holding_period_bars=2, periods_per_year=1)
    result = evaluator.evaluate("test", weights, ohlcv, funding=funding)

    assert result.returns[0] == pytest.approx(+0.001, abs=1e-9)
    assert result.total_funding_cost == pytest.approx(-0.001, abs=1e-12)


def test_funding_sums_all_events_in_holding_period() -> None:
    """Two funding events inside the holding period are summed."""
    ohlcv = _ohlcv([100.0] * 8)
    ts = ohlcv["timestamp"].to_list()
    weights = _weights([ts[0]], [1.0])
    # h=4 → interval (ts[1], ts[5]] contains ts[2], ts[3], ts[4], ts[5]
    funding = _funding([ts[2], ts[3], ts[4], ts[5]], [0.001, 0.001, 0.001, 0.001])

    evaluator = VectorizedEvaluator(holding_period_bars=4, periods_per_year=1)
    result = evaluator.evaluate("test", weights, ohlcv, funding=funding)

    assert result.n_funding_events == 4
    assert result.returns[0] == pytest.approx(-0.004, abs=1e-9)


def test_funding_event_at_exec_ts_is_excluded() -> None:
    """Events exactly at exec_ts belong to the previous period."""
    ohlcv = _ohlcv([100.0, 100.0, 100.0, 100.0])
    ts = ohlcv["timestamp"].to_list()
    weights = _weights([ts[0]], [1.0])
    # h=1 → interval (ts[1], ts[2]]; ts[1] must be excluded
    funding = _funding([ts[1], ts[2]], [0.001, 0.001])

    evaluator = VectorizedEvaluator(holding_period_bars=1, periods_per_year=1)
    result = evaluator.evaluate("test", weights, ohlcv, funding=funding)

    assert result.n_funding_events == 1
    assert result.returns[0] == pytest.approx(-0.001, abs=1e-9)


def test_no_funding_is_noop() -> None:
    ohlcv = _ohlcv([100.0, 110.0, 121.0])
    ts = ohlcv["timestamp"].to_list()
    weights = _weights([ts[0]], [1.0])

    evaluator = VectorizedEvaluator(holding_period_bars=1, periods_per_year=1)
    result = evaluator.evaluate("test", weights, ohlcv)

    assert result.returns[0] == pytest.approx(0.1, abs=1e-9)
    assert result.total_funding_cost == 0.0
    assert result.n_funding_events == 0
