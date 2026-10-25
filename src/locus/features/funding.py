"""
funding.py

Funding rate features. All computed on a DataFrame with the funding
schema (asset_id, venue, symbol, timestamp, funding_rate).

Point-in-time: at timestamp t, features use only funding data with
timestamp <= t. Enforced by tests/unit/features/test_funding_leakage.py.

Null contract (z-score features):
    - None during warm-up (incomplete rolling window) or where the input is null.
    - 0.0 when the rolling / cross-sectional standard deviation is exactly zero.
      A constant rate carries no dispersion information; returning 0.0 (neutral)
      avoids a selection bias: calm markets (funding pinned at the venue default,
      e.g. Binance 0.0001) would otherwise be systematically excluded when nulls
      are dropped before training. Observed empirically: ~6% of 7d z-scores on
      a 5-asset universe were nulls concentrated in calm markets before this fix.
    - the z-score otherwise.

Timestamp alignment: funding timestamps carry 2-3ms drift in
production (observed at 2026-09-30 16:00:00.002+00:00). Cross-
sectional features group by timestamp and require alignment across
assets. `normalize_funding_timestamps` floors each timestamp to the
nearest minute, absorbing the drift without touching the values.

Author: Marco Pérez Padilla
Date:   24-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from locus.features.base import register_feature


def normalize_funding_timestamps(
    df: pl.DataFrame,
    rounding: Literal["1s", "1m", "1h"] = "1m",
) -> pl.DataFrame:
    """Floor funding timestamps to a fixed unit to absorb ms drift.

    Funding events land at 00:00, 08:00, 16:00 UTC with 2-3ms drift.
    Flooring to 1 minute collapses the drift. Cross-sectional features
    (which group by timestamp) must be computed on normalized data;
    per-asset features are agnostic (they group by asset_id).

    NOTE: this function returns a new DataFrame where the `timestamp`
    column is floored. Callers that need the original timestamp for
    downstream joins (e.g. with OHLCV bars) must preserve it separately.
    """
    return df.with_columns(pl.col("timestamp").dt.truncate(rounding).alias("timestamp"))


@dataclass(frozen=True, slots=True)
class FundingZScoreFeature:
    """Rolling z-score of funding_rate over `window` periods, per asset.

    Null contract:
        - None during warm-up or where `funding_rate` is null.
        - 0.0 when rolling std is exactly 0 (constant rate in the window).
        - z-score otherwise.
    """

    name: str
    window: int = 90

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be >= 2")

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        if "funding_rate" not in df.columns:
            raise ValueError(f"{self.name!r} requires column 'funding_rate'")
        result = df.sort(["asset_id", "timestamp"])
        mean = pl.col("funding_rate").rolling_mean(self.window).over("asset_id")
        std = pl.col("funding_rate").rolling_std(self.window).over("asset_id")
        return result.with_columns(
            pl.when(pl.col("funding_rate").is_null() | std.is_null())
            .then(None)
            .when(std == 0)
            .then(0.0)
            .otherwise((pl.col("funding_rate") - mean) / std)
            .alias(self.name)
        )


@dataclass(frozen=True, slots=True)
class FundingExtremeFeature:
    """Binary flag: 1 if |source| > threshold, 0 otherwise. Null where
    source is null.

    Note: since z-score features return 0.0 (not null) when std == 0, an
    extreme flag built on top of them inherits 0 (not extreme) in those rows.
    This is intentional: a constant funding rate is not an extreme event.
    """

    name: str
    source: str = "funding_zscore_30d"
    threshold: float = 2.0

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError("threshold must be > 0")

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.source not in df.columns:
            raise ValueError(f"{self.name!r} requires column {self.source!r}")
        return df.with_columns(
            pl.when(pl.col(self.source).is_not_null())
            .then((pl.col(self.source).abs() > self.threshold).cast(pl.Int8))
            .otherwise(None)
            .alias(self.name)
        )


@dataclass(frozen=True, slots=True)
class FundingTrendFeature:
    """Change in funding_rate over `window` periods:
    funding_rate[t] - funding_rate[t-window]. Null during warm-up."""

    name: str
    window: int = 21

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        if "funding_rate" not in df.columns:
            raise ValueError(f"{self.name!r} requires column 'funding_rate'")
        result = df.sort(["asset_id", "timestamp"])
        return result.with_columns(
            (
                pl.col("funding_rate") - pl.col("funding_rate").shift(self.window).over("asset_id")
            ).alias(self.name)
        )


@dataclass(frozen=True, slots=True)
class FundingXSRankFeature:
    """Cross-sectional percentile rank of `source` across assets at
    each timestamp, in (0, 1). Ties get the same value. Normalizes
    timestamps before grouping to absorb drift.

    NOTE: the output DataFrame carries the floored timestamp, not the
    original. If downstream joins with OHLCV require the original
    timestamp, join before this feature or preserve a copy of the
    timestamp column."""

    name: str
    source: str = "funding_rate"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.source not in df.columns:
            raise ValueError(f"{self.name!r} requires column {self.source!r}")
        normalized = normalize_funding_timestamps(df)
        non_null_count = pl.col(self.source).count().over("timestamp")
        return normalized.with_columns(
            pl.when(pl.col(self.source).is_not_null())
            .then(
                (pl.col(self.source).rank(method="average").over("timestamp") - 0.5)
                / non_null_count
            )
            .otherwise(None)
            .alias(self.name)
        )


@dataclass(frozen=True, slots=True)
class FundingXSZScoreFeature:
    """Cross-sectional z-score of `source` across assets at each
    timestamp.

    Null contract (mirrors FundingZScoreFeature):
        - None when `source` is null or fewer than 2 assets in the cross-section.
        - 0.0 when cross-sectional std is exactly 0 (all assets share the rate).
        - z-score otherwise.

    NOTE: the output DataFrame carries the floored timestamp, not the
    original (see FundingXSRankFeature)."""

    name: str
    source: str = "funding_rate"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.source not in df.columns:
            raise ValueError(f"{self.name!r} requires column {self.source!r}")
        normalized = normalize_funding_timestamps(df)
        mean = pl.col(self.source).mean().over("timestamp")
        std = pl.col(self.source).std().over("timestamp")
        return normalized.with_columns(
            pl.when(pl.col(self.source).is_null() | std.is_null())
            .then(None)
            .when(std == 0)
            .then(0.0)
            .otherwise((pl.col(self.source) - mean) / std)
            .alias(self.name)
        )


@register_feature("funding_zscore")
def _make_funding_zscore(name: str, window: int = 90) -> FundingZScoreFeature:
    return FundingZScoreFeature(name=name, window=window)


@register_feature("funding_extreme")
def _make_funding_extreme(
    name: str,
    source: str = "funding_zscore_30d",
    threshold: float = 2.0,
) -> FundingExtremeFeature:
    return FundingExtremeFeature(name=name, source=source, threshold=threshold)


@register_feature("funding_trend")
def _make_funding_trend(name: str, window: int = 21) -> FundingTrendFeature:
    return FundingTrendFeature(name=name, window=window)


@register_feature("funding_xs_rank")
def _make_funding_xs_rank(name: str, source: str = "funding_rate") -> FundingXSRankFeature:
    return FundingXSRankFeature(name=name, source=source)


@register_feature("funding_xs_zscore")
def _make_funding_xs_zscore(name: str, source: str = "funding_rate") -> FundingXSZScoreFeature:
    return FundingXSZScoreFeature(name=name, source=source)
