"""
test_extended.py

Extended per-asset features: numerical correctness, nulls at the
start, per-asset grouping.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.data.providers.base import BAR_SCHEMA
from locus.features import (
    ATRPercentFeature,
    EMADistanceFeature,
    RelativeVolumeFeature,
    VolatilityRatioFeature,
    VolumeZScoreFeature,
)


def _df(
    closes: list[float],
    volumes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    vols = volumes or [1.0] * len(closes)
    hs = highs or [c * 1.01 for c in closes]
    ls = lows or [c * 0.99 for c in closes]
    return pl.DataFrame(
        [
            {
                "asset_id": "A",
                "venue": "test",
                "symbol": "AUSDT",
                "timeframe": "1h",
                "timestamp": base + timedelta(hours=i),
                "open": closes[i],
                "high": hs[i],
                "low": ls[i],
                "close": closes[i],
                "volume": vols[i],
            }
            for i in range(len(closes))
        ],
        schema=BAR_SCHEMA,
    )


def test_ema_distance_numerically_sane() -> None:
    df = _df([100.0, 100.0, 100.0, 120.0])
    result = EMADistanceFeature(name="ema_distance_3", window=3).compute(df)
    values = result.sort("timestamp")["ema_distance_3"].to_list()
    # EMA warms up, but after 4 bars the value is defined and positive
    assert values[3] is not None
    assert values[3] > 0


def test_ema_distance_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError):
        EMADistanceFeature(name="x", window=0)


def test_atr_pct_zero_on_flat_bar() -> None:
    # Flat bars: high-low = 0, |high-prev_close| = 0, |low-prev_close| = 0
    closes = [100.0] * 10
    highs = [100.0] * 10
    lows = [100.0] * 10
    df = _df(closes, highs=highs, lows=lows)
    result = ATRPercentFeature(name="atr_pct_3", window=3).compute(df)
    values = result.sort("timestamp")["atr_pct_3"].to_list()
    assert values[5] == pytest.approx(0.0)


def test_atr_pct_nulls_at_start() -> None:
    df = _df([100.0, 101.0, 102.0, 103.0])
    result = ATRPercentFeature(name="atr_pct_2", window=2).compute(df)
    values = result.sort("timestamp")["atr_pct_2"].to_list()
    # window=2 → first defined at index 1
    assert values[0] is None
    assert values[1] is not None


def test_atr_pct_rejects_small_window() -> None:
    with pytest.raises(ValueError):
        ATRPercentFeature(name="x", window=1)


def test_volatility_ratio_null_when_long_vol_zero() -> None:
    # Constant price → zero volatility → null ratio
    df = _df([100.0] * 30)
    result = VolatilityRatioFeature(name="vr_5_20", short_window=5, long_window=20).compute(df)
    values = result.sort("timestamp")["vr_5_20"].to_list()
    # Long vol is 0 in the constant series, so ratio is null throughout
    assert all(v is None for v in values[20:])


def test_volatility_ratio_rejects_invalid_windows() -> None:
    with pytest.raises(ValueError):
        VolatilityRatioFeature(name="x", short_window=1, long_window=20)
    with pytest.raises(ValueError):
        VolatilityRatioFeature(name="x", short_window=10, long_window=10)


def test_volume_zscore_null_on_constant_volume() -> None:
    df = _df([100.0 + i for i in range(20)], volumes=[10.0] * 20)
    result = VolumeZScoreFeature(name="vz_5", window=5).compute(df)
    values = result.sort("timestamp")["vz_5"].to_list()
    # Constant volume → std = 0 → null
    assert all(v is None for v in values[5:])


def test_volume_zscore_positive_on_spike() -> None:
    vols = [10.0] * 10 + [100.0] + [10.0] * 5
    df = _df([100.0 + i for i in range(len(vols))], volumes=vols)
    result = VolumeZScoreFeature(name="vz_5", window=5).compute(df)
    values = result.sort("timestamp")["vz_5"].to_list()
    # Row 10 is a spike relative to rows 5-9
    assert values[10] is not None
    assert values[10] > 0


def test_relative_volume_above_one_on_spike() -> None:
    vols = [10.0] * 10 + [100.0] + [10.0] * 5
    df = _df([100.0 + i for i in range(len(vols))], volumes=vols)
    result = RelativeVolumeFeature(name="rv_5", window=5).compute(df)
    values = result.sort("timestamp")["rv_5"].to_list()
    assert values[10] is not None
    assert values[10] > 1.0


def test_relative_volume_rejects_small_window() -> None:
    with pytest.raises(ValueError):
        RelativeVolumeFeature(name="x", window=0)
