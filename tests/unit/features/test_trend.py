"""
test_trend.py

MovingAverageDistanceFeature: numerical correctness, nulls at the start.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.data.providers.base import BAR_SCHEMA
from locus.features import MovingAverageDistanceFeature


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


def test_ma_distance_numerically_correct() -> None:
    df = _df([100.0, 100.0, 100.0, 120.0])
    result = (
        MovingAverageDistanceFeature(name="ma_distance_2", window=2).compute(df).sort("timestamp")
    )
    values = result["ma_distance_2"].to_list()
    assert values[0] is None
    assert values[1] == pytest.approx(0.0)
    assert values[2] == pytest.approx(0.0)
    assert values[3] == pytest.approx(10.0 / 110.0)


def test_ma_distance_rejects_small_window() -> None:
    with pytest.raises(ValueError):
        MovingAverageDistanceFeature(name="x", window=0)
