"""
test_leakage.py

The most important tests in F3. Features at timestamp t must depend
only on data at timestamps <= t within the same asset. If a feature
uses future data (look-ahead bias), these tests fail.

Strategy: compute features on the full history, then compute them on
a history truncated at `cutoff`, and verify the values at timestamps
<= cutoff are identical.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from locus.data.providers.base import BAR_SCHEMA
from locus.features import (
    FeaturePipeline,
    LogReturnFeature,
    MovingAverageDistanceFeature,
    ReturnFeature,
    RollingVolatilityFeature,
)


def _df(n: int = 100) -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for asset_id in ("A", "B"):
        for i in range(n):
            price = 100.0 + i * 0.5 + (10.0 if asset_id == "B" else 0.0)
            rows.append(
                {
                    "asset_id": asset_id,
                    "venue": "test",
                    "symbol": f"{asset_id}USDT",
                    "timeframe": "1h",
                    "timestamp": base + timedelta(hours=i),
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1.0 + 0.1 * i,
                }
            )
    return pl.DataFrame(rows, schema=BAR_SCHEMA)


def _pipeline() -> FeaturePipeline:
    return FeaturePipeline(
        [
            ReturnFeature(name="return_1", n=1),
            ReturnFeature(name="return_5", n=5),
            LogReturnFeature(name="log_return_1", n=1),
            MovingAverageDistanceFeature(name="ma_distance_20", window=20),
            RollingVolatilityFeature(name="rolling_vol_20", window=20),
        ]
    )


def test_no_leakage_under_future_truncation() -> None:
    df = _df()
    cutoff = df["timestamp"].max() - timedelta(hours=10)

    full_features = _pipeline().compute(df)
    full_truncated = full_features.filter(pl.col("timestamp") <= cutoff).sort(
        ["asset_id", "timestamp"]
    )

    truncated_input = df.filter(pl.col("timestamp") <= cutoff)
    truncated_features = _pipeline().compute(truncated_input).sort(["asset_id", "timestamp"])

    feature_cols = _pipeline().feature_names
    assert full_truncated.select(feature_cols).equals(truncated_features.select(feature_cols)), (
        "features at t <= cutoff differ when future is truncated (leakage)"
    )
