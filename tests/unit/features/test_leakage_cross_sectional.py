"""
test_leakage_cross_sectional.py

Cross-sectional leakage tests. Asset A's cross-sectional feature at
timestamp t must not depend on asset B's future data. Multi-asset
panel with truncation invariance.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from locus.data.providers.base import BAR_SCHEMA
from locus.features import (
    CrossSectionalPercentileRankFeature,
    CrossSectionalZScoreFeature,
    FeaturePipeline,
    RelativeMomentumVsMarketFeature,
    ReturnFeature,
)


def _panel(n_bars: int = 50, assets: tuple[str, ...] = ("A", "B", "C")) -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for i, asset_id in enumerate(assets):
        # Different price dynamics per asset so cross-section differs
        base_price = 100.0 + 10.0 * i
        for t in range(n_bars):
            price = base_price * (1.0 + 0.01 * ((t + i) % 5) + 0.001 * t)
            rows.append(
                {
                    "asset_id": asset_id,
                    "venue": "test",
                    "symbol": f"{asset_id}USDT",
                    "timeframe": "1h",
                    "timestamp": base + timedelta(hours=t),
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1.0 + 0.1 * t + 0.5 * i,
                }
            )
    return pl.DataFrame(rows, schema=BAR_SCHEMA)


def _pipeline() -> FeaturePipeline:
    return FeaturePipeline(
        [
            ReturnFeature(name="return_5", n=5),
            ReturnFeature(name="return_10", n=10),
            CrossSectionalPercentileRankFeature(name="xs_rank_r5", source="return_5"),
            CrossSectionalZScoreFeature(name="xs_z_r5", source="return_5"),
            CrossSectionalPercentileRankFeature(name="xs_rank_r10", source="return_10"),
            RelativeMomentumVsMarketFeature(name="rel_mom_r5", source="return_5"),
        ]
    )


def test_no_leakage_cross_sectional_under_truncation() -> None:
    df = _panel()
    cutoff = df["timestamp"].max() - timedelta(hours=10)

    full_features = _pipeline().compute(df)
    full_truncated = full_features.filter(pl.col("timestamp") <= cutoff).sort(
        ["asset_id", "timestamp"]
    )

    truncated_input = df.filter(pl.col("timestamp") <= cutoff)
    truncated_features = _pipeline().compute(truncated_input).sort(["asset_id", "timestamp"])

    feature_cols = _pipeline().feature_names
    assert full_truncated.select(feature_cols).equals(truncated_features.select(feature_cols)), (
        "cross-sectional features at t <= cutoff differ when the future "
        "is truncated (cross-asset leakage)"
    )


def test_no_cross_asset_contamination_from_anomaly() -> None:
    """Inject a massive price spike in asset A at the last bar. Asset
    B's cross-sectional features at earlier timestamps must be
    unaffected."""
    df = _panel()
    last_ts = df["timestamp"].max()
    anomaly = df.with_columns(
        pl.when((pl.col("asset_id") == "A") & (pl.col("timestamp") == last_ts))
        .then(pl.col("close") * 1000.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )

    feature_cols = [
        "xs_rank_r5",
        "xs_z_r5",
        "rel_mom_r5",
    ]
    normal_features = _pipeline().compute(df)
    anomaly_features = _pipeline().compute(anomaly)

    cutoff = last_ts - timedelta(hours=2)
    normal_b_before = (
        normal_features.filter((pl.col("asset_id") == "B") & (pl.col("timestamp") <= cutoff))
        .sort("timestamp")
        .select(feature_cols)
    )
    anomaly_b_before = (
        anomaly_features.filter((pl.col("asset_id") == "B") & (pl.col("timestamp") <= cutoff))
        .sort("timestamp")
        .select(feature_cols)
    )

    assert normal_b_before.equals(anomaly_b_before), (
        "asset A's future anomaly affected asset B's cross-sectional "
        "features at earlier timestamps (cross-asset leakage)"
    )
