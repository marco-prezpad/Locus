"""
test_funding.py

Funding feature correctness: numerical values, warm-up nulls, cross-
sectional alignment with drift, isolated-asset behavior.

Author: Marco Pérez Padilla
Date:   24-10-2026
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.data.providers.funding_base import FUNDING_SCHEMA
from locus.features.funding import (
    FundingExtremeFeature,
    FundingTrendFeature,
    FundingXSRankFeature,
    FundingXSZScoreFeature,
    FundingZScoreFeature,
    normalize_funding_timestamps,
)


def _multi_asset_series(
    series_by_asset: dict[str, list[float]],
    *,
    step_hours: int = 8,
) -> pl.DataFrame:
    """Build a funding DataFrame with one asset_id per key.

    All series must have the same length; funding events are assumed to
    be aligned across assets (00:00, 08:00, 16:00 UTC by default).
    Only the columns the funding features actually need are populated
    (asset_id, timestamp, funding_rate).
    """
    lengths = {len(v) for v in series_by_asset.values()}
    if len(lengths) != 1:
        raise ValueError("all series must have the same length")
    n = lengths.pop()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(hours=step_hours * i) for i in range(n)]
    return pl.DataFrame(
        [
            {"asset_id": asset_id, "timestamp": ts, "funding_rate": rate}
            for asset_id, values in series_by_asset.items()
            for ts, rate in zip(timestamps, values, strict=False)
        ]
    )


def _series(values: list[float], asset: str = "A") -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {
            "asset_id": asset,
            "venue": "binance_futures",
            "symbol": f"{asset}USDT",
            "timestamp": base + timedelta(hours=8 * i),
            "funding_rate": float(v),
        }
        for i, v in enumerate(values)
    ]
    return pl.DataFrame(rows, schema=FUNDING_SCHEMA)


def test_zscore_numerically_correct() -> None:
    # Known series: mean of last 3 = (0.0002+0.0003+0.0004)/3 = 0.0003
    values = [0.0001, 0.0002, 0.0003, 0.0004]
    df = _series(values)
    result = FundingZScoreFeature(name="z", window=3).compute(df)
    z = result["z"].to_list()
    assert z[0] is None and z[1] is None
    assert z[2] is not None
    import numpy as np

    expected = (0.0003 - np.mean(values[:3])) / np.std(values[:3], ddof=1)
    assert z[2] == pytest.approx(expected, rel=1e-8)


def test_zscore_nulls_during_warmup() -> None:
    # Non-constant data so that post-warm-up std > 0: the only source of
    # null is the incomplete window.
    values = [0.0001 if i % 2 == 0 else 0.0002 for i in range(100)]
    df = _series(values)
    result = FundingZScoreFeature(name="z", window=21).compute(df)
    z = result.sort("timestamp")["z"].to_list()
    assert all(v is None for v in z[:20])
    assert z[20] is not None
    assert not math.isnan(z[20])


def test_zscore_is_zero_when_std_is_zero() -> None:
    # Constant funding rate: no dispersion, so the z-score must be the
    # neutral 0.0 (not null). Guards against the selection bias described
    # in the feature docstring.
    df = _series([0.0001] * 100)
    result = FundingZScoreFeature(name="z", window=21).compute(df)
    z = result.sort("timestamp")["z"].to_list()
    assert all(v is None for v in z[:20])
    assert all(v == 0.0 for v in z[20:])


def test_xs_zscore_is_zero_when_std_is_zero() -> None:
    # All assets share the same funding rate at each timestamp → no
    # cross-sectional dispersion → neutral 0.0.
    df = _multi_asset_series(
        {
            "BTC": [0.0001] * 50,
            "ETH": [0.0001] * 50,
            "SOL": [0.0001] * 50,
        }
    )
    result = FundingXSZScoreFeature(name="xs_z").compute(df)
    z = result.sort(["timestamp", "asset_id"])["xs_z"].to_list()
    assert all(v == 0.0 for v in z)


def test_zscore_null_when_std_is_zero() -> None:
    df = _series([0.0001] * 100)
    result = FundingZScoreFeature(name="z", window=21).compute(df)
    z = result.sort("timestamp")["z"].to_list()
    assert all(v is None for v in z[:20])
    assert all(v == 0.0 for v in z[20:])


def test_zscore_null_on_constant_series() -> None:
    df = _series([0.0001] * 30)
    result = FundingZScoreFeature(name="z", window=10).compute(df)
    z = result.sort("timestamp")["z"].to_list()
    assert all(v is None for v in z[:9])
    assert all(v == 0.0 for v in z[9:])


def test_zscore_rejects_short_window() -> None:
    with pytest.raises(ValueError):
        FundingZScoreFeature(name="z", window=1)


def test_zscore_requires_funding_rate_column() -> None:
    with pytest.raises(ValueError, match="funding_rate"):
        FundingZScoreFeature(name="z", window=5).compute(pl.DataFrame({"asset_id": ["A"]}))


def test_extreme_flag_correctness() -> None:
    df = _series([0.0001] * 20).with_columns(pl.Series("z", [0.5, 1.5, 2.5, -2.5] + [None] * 16))
    result = FundingExtremeFeature(name="ext", source="z", threshold=2.0).compute(df)
    ext = result.sort("timestamp")["ext"].to_list()
    assert ext[0] == 0
    assert ext[1] == 0
    assert ext[2] == 1
    assert ext[3] == 1
    assert all(v is None for v in ext[4:])


def test_trend_correctness() -> None:
    values = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
    df = _series(values)
    result = FundingTrendFeature(name="t", window=2).compute(df)
    t = result.sort("timestamp")["t"].to_list()
    assert t[0] is None and t[1] is None
    assert t[2] == pytest.approx(0.0002)
    assert t[4] == pytest.approx(0.0002)


def test_xs_rank_aligned_with_drift() -> None:
    """Two assets with different ms drift must align after normalization."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {
            "asset_id": "A",
            "venue": "binance_futures",
            "symbol": "AUSDT",
            "timestamp": base + timedelta(milliseconds=2),
            "funding_rate": 0.0001,
        },
        {
            "asset_id": "B",
            "venue": "binance_futures",
            "symbol": "BUSDT",
            "timestamp": base + timedelta(milliseconds=4),
            "funding_rate": 0.0002,
        },
    ]
    df = pl.DataFrame(rows, schema=FUNDING_SCHEMA)
    result = FundingXSRankFeature(name="r", source="funding_rate").compute(df)
    ranks = result.sort("asset_id")["r"].to_list()
    assert ranks[0] == pytest.approx(0.25)  # A: lower rate → lower rank
    assert ranks[1] == pytest.approx(0.75)


def test_xs_zscore_aligned_with_drift() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {
            "asset_id": "A",
            "venue": "binance_futures",
            "symbol": "AUSDT",
            "timestamp": base + timedelta(milliseconds=2),
            "funding_rate": 0.0001,
        },
        {
            "asset_id": "B",
            "venue": "binance_futures",
            "symbol": "BUSDT",
            "timestamp": base + timedelta(milliseconds=4),
            "funding_rate": 0.0003,
        },
    ]
    df = pl.DataFrame(rows, schema=FUNDING_SCHEMA)
    result = FundingXSZScoreFeature(name="zs", source="funding_rate").compute(df)
    zs = result.sort("asset_id")["zs"].to_list()
    # mean = 0.0002, std (ddof=1) = 0.0001414
    assert zs[0] == pytest.approx(-0.7071, abs=1e-3)
    assert zs[1] == pytest.approx(0.7071, abs=1e-3)


def test_xs_rank_single_asset_returns_half() -> None:
    df = _series([0.0001] * 5)
    result = FundingXSRankFeature(name="r", source="funding_rate").compute(df)
    ranks = result["r"].to_list()
    assert all(v == pytest.approx(0.5) for v in ranks)


def test_xs_zscore_single_asset_returns_null() -> None:
    df = _series([0.0001] * 5)
    result = FundingXSZScoreFeature(name="zs", source="funding_rate").compute(df)
    zs = result["zs"].to_list()
    assert all(v is None for v in zs)


def test_normalize_idempotent() -> None:
    df = _series([0.0001] * 5)
    once = normalize_funding_timestamps(df)
    twice = normalize_funding_timestamps(once)
    assert once.equals(twice)
