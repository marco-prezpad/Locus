"""
test_market_state.py

Market state estimation: sigma_daily, daily_volume_quote, nulls at
start, per-asset grouping, numerical sanity.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.backtest.market_state import (
    enrich_with_market_state,
    estimate_daily_volume_quote,
    estimate_sigma_daily,
)
from locus.data.providers.base import BAR_SCHEMA


def _df(n: int = 50, asset: str = "A") -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "asset_id": asset,
                "venue": "t",
                "symbol": f"{asset}USDT",
                "timeframe": "1h",
                "timestamp": base + timedelta(hours=i),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 + 0.1 * i,
                "volume": 10.0 + 0.01 * i,
            }
            for i in range(n)
        ],
        schema=BAR_SCHEMA,
    )


def test_sigma_daily_added_and_nulls_at_start() -> None:
    df = _df(50)
    result = estimate_sigma_daily(df, bars_per_day=24, window=24)
    assert "sigma_daily" in result.columns
    # First row null (no returns); rows 1..24 null (rolling window=24)
    sigmas = result.sort("timestamp")["sigma_daily"].to_list()
    assert sigmas[0] is None
    assert sigmas[23] is None
    assert sigmas[24] is not None


def test_sigma_daily_scales_with_bars_per_day() -> None:
    df = _df(50)
    s1 = estimate_sigma_daily(df, bars_per_day=24, window=10).sort("timestamp")
    s2 = estimate_sigma_daily(df, bars_per_day=96, window=10).sort("timestamp")
    # 96 = 4 * 24, so sqrt(4) = 2x
    v1 = s1["sigma_daily"][20]
    v2 = s2["sigma_daily"][20]
    assert v2 == pytest.approx(2.0 * v1, rel=1e-9)


def test_daily_volume_quote_added() -> None:
    df = _df(50)
    result = estimate_daily_volume_quote(df, bars_per_day=24, window=24)
    assert "daily_volume_quote" in result.columns
    # After warm-up, positive
    vols = result.sort("timestamp")["daily_volume_quote"].to_list()
    assert vols[30] is not None
    assert vols[30] > 0


def test_daily_volume_scales_with_bars_per_day() -> None:
    df = _df(50)
    v1 = estimate_daily_volume_quote(df, bars_per_day=24, window=10).sort("timestamp")
    v2 = estimate_daily_volume_quote(df, bars_per_day=48, window=10).sort("timestamp")
    # 2x bars_per_day → 2x daily volume
    assert v2["daily_volume_quote"][20] == pytest.approx(
        2.0 * v1["daily_volume_quote"][20], rel=1e-9
    )


def test_enrich_adds_both_columns() -> None:
    df = _df(50)
    result = enrich_with_market_state(df)
    assert "sigma_daily" in result.columns
    assert "daily_volume_quote" in result.columns


def test_per_asset_isolation() -> None:
    a = _df(50, "A")
    b = _df(50, "B").with_columns(pl.col("close") * 2.0)
    df = pl.concat([a, b])
    result = estimate_sigma_daily(df, bars_per_day=24, window=10)
    # No cross-contamination: A's sigma at row 20 ≈ B's sigma at row 20
    # only if their return series are identical. Here B is 2x A, so
    # returns are identical → sigmas identical.
    a_sig = result.filter(pl.col("asset_id") == "A").sort("timestamp")["sigma_daily"][20]
    b_sig = result.filter(pl.col("asset_id") == "B").sort("timestamp")["sigma_daily"][20]
    assert a_sig == pytest.approx(b_sig, rel=1e-9)


def test_rejects_invalid_params() -> None:
    df = _df(10)
    with pytest.raises(ValueError):
        estimate_sigma_daily(df, bars_per_day=0)
    with pytest.raises(ValueError):
        estimate_sigma_daily(df, window=1)
    with pytest.raises(ValueError):
        estimate_daily_volume_quote(df, bars_per_day=0)
