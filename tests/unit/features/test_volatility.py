"""
test_volatility.py

RollingVolatilityFeature: numerical correctness on constant returns,
nulls at the start, per-asset grouping.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.data.providers.base import BAR_SCHEMA
from locus.features import RollingVolatilityFeature


def _df(closes: list[float]) -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "asset_id": "A",
                "venue": "test",
                "symbol": "AUSDT",
                "timeframe": "1h",
                "timestamp": base + timedelta(hours=i),
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "volume": 1.0,
            }
            for i, c in enumerate(closes)
        ],
        schema=BAR_SCHEMA,
    )


def test_rolling_vol_zero_on_constant_returns() -> None:
    df = _df([100.0, 110.0, 121.0, 133.1, 146.41])
    result = RollingVolatilityFeature(name="rolling_vol_3", window=3).compute(df).sort("timestamp")
    values = result["rolling_vol_3"].to_list()
    assert values[3] == pytest.approx(0.0)
    assert values[4] == pytest.approx(0.0)


def test_rolling_vol_nulls_at_start() -> None:
    df = _df([100.0, 101.0, 102.0, 103.0])
    result = RollingVolatilityFeature(name="rolling_vol_2", window=2).compute(df).sort("timestamp")
    values = result["rolling_vol_2"].to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] is not None
    assert values[3] is not None


def test_rolling_vol_rejects_window_below_2() -> None:
    with pytest.raises(ValueError):
        RollingVolatilityFeature(name="x", window=1)
