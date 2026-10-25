"""
test_funding_leakage.py

Truncation-invariance test for funding features. Per pre-registration
§6.3, this test is written FIRST, before any feature implementation.
If it fails once features exist, there is leakage and they are not
shipped.

Strategy: compute features on the full funding history, truncate at
`cutoff`, recompute on the truncated input, assert values at
`t <= cutoff` are identical. Any feature that peeks at future funding
(shift(-n), center=True, forward-fill from future, future row in a
cross-sectional group) fails.

Author: Marco Pérez Padilla
Date:   24-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from locus.data.providers.funding_base import FUNDING_SCHEMA
from locus.features import FeaturePipeline, FeatureSpec
from locus.features.funding import normalize_funding_timestamps


def _funding_df(
    n_periods: int = 200,
    assets: tuple[str, ...] = ("A", "B"),
) -> pl.DataFrame:
    """Synthetic funding data: 8h cadence, deterministic values, and
    2-3ms drift matching production (observed at 2026-09-30 16:00:00.002)."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for i, asset in enumerate(assets):
        for t in range(n_periods):
            ts = base + timedelta(hours=8 * t, milliseconds=2 * (i + 1))
            rate = 0.0001 * ((t + i) % 5 - 2)
            rows.append(
                {
                    "asset_id": asset,
                    "venue": "binance_futures",
                    "symbol": f"{asset}USDT",
                    "timestamp": ts,
                    "funding_rate": float(rate),
                }
            )
    return pl.DataFrame(rows, schema=FUNDING_SCHEMA)


def _pipeline() -> FeaturePipeline:
    return FeaturePipeline.from_specs(
        [
            FeatureSpec(name="funding_zscore_30d", family="funding_zscore", params={"window": 90}),
            FeatureSpec(name="funding_zscore_7d", family="funding_zscore", params={"window": 21}),
            FeatureSpec(
                name="funding_extreme_30d",
                family="funding_extreme",
                params={"source": "funding_zscore_30d"},
            ),
            FeatureSpec(name="funding_trend_7d", family="funding_trend", params={"window": 21}),
            FeatureSpec(
                name="funding_xs_rank", family="funding_xs_rank", params={"source": "funding_rate"}
            ),
            FeatureSpec(
                name="funding_xs_zscore",
                family="funding_xs_zscore",
                params={"source": "funding_rate"},
            ),
        ]
    )


def test_funding_no_lookahead_under_truncation() -> None:
    df = normalize_funding_timestamps(_funding_df())
    cutoff = df["timestamp"].unique().sort()[149]

    full = _pipeline().compute(df)
    truncated = _pipeline().compute(df.filter(pl.col("timestamp") <= cutoff))

    feature_cols = _pipeline().feature_names
    full_at_cutoff = full.filter(pl.col("timestamp") <= cutoff).sort(["asset_id", "timestamp"])
    truncated_sorted = truncated.sort(["asset_id", "timestamp"])

    for col in feature_cols:
        assert full_at_cutoff[col].equals(truncated_sorted[col]), (
            f"feature {col!r} at t <= cutoff differs when the future is truncated: leakage detected"
        )
