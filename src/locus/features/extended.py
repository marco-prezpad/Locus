"""
extended.py

Extended per-asset features: EMA distance, ATR as percentage of close,
volatility ratio (short vs long window), volume z-score, and relative
volume.

All follow the same backward-looking contract as the F3.1 features:
at row t, values depend only on data at timestamps <= t within the
same asset.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from locus.features.base import register_feature


@dataclass(frozen=True, slots=True)
class EMADistanceFeature:
    """Distance from close to an exponential moving average:
    (close - EMA) / EMA.

    EMA with span=window weights recent observations more heavily than
    the simple moving average. Null for the first rows of each asset
    (EWM warm-up), roughly `window - 1` nulls.
    """

    name: str
    window: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.window <= 0:
            raise ValueError("window must be positive")

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        df_sorted = df.sort(["asset_id", "timestamp"])
        with_ema = df_sorted.with_columns(
            pl.col("close").ewm_mean(span=self.window, adjust=False).over("asset_id").alias("_ema")
        )
        return with_ema.with_columns(
            ((pl.col("close") - pl.col("_ema")) / pl.col("_ema")).alias(self.name)
        ).drop("_ema")


@dataclass(frozen=True, slots=True)
class ATRPercentFeature:
    """Average True Range as a fraction of close:
    atr_w[t] / close[t].

    True Range = max(high - low, |high - prev_close|, |low - prev_close|).
    ATR = rolling mean of TR over `window` bars. Dividing by close makes
    the value dimensionless (comparable across assets and timeframes).

    Null for the first `window - 1` rows of each asset.
    """

    name: str
    window: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.window < 2:
            raise ValueError("window must be >= 2")

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        df_sorted = df.sort(["asset_id", "timestamp"])
        prev_close = pl.col("close").shift(1).over("asset_id")
        tr = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - prev_close).abs().fill_null(0.0),
            (pl.col("low") - prev_close).abs().fill_null(0.0),
        )
        with_tr = df_sorted.with_columns(tr.alias("_tr"))
        with_atr = with_tr.with_columns(
            pl.col("_tr").rolling_mean(self.window).over("asset_id").alias("_atr")
        )
        return with_atr.with_columns((pl.col("_atr") / pl.col("close")).alias(self.name)).drop(
            ["_tr", "_atr"]
        )


@dataclass(frozen=True, slots=True)
class VolatilityRatioFeature:
    """Ratio of short-window to long-window rolling volatility.

    Values > 1 indicate recent volatility above the longer-term level
    (vol regime shift). Values < 1 indicate the opposite.

    Null for the first `long_window` rows of each asset. Null also when
    the long-window volatility is exactly 0 (flat price), since the
    ratio would be undefined.
    """

    name: str
    short_window: int
    long_window: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.short_window < 2:
            raise ValueError("short_window must be >= 2")
        if self.long_window <= self.short_window:
            raise ValueError("long_window must be > short_window")

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        df_sorted = df.sort(["asset_id", "timestamp"])
        with_ret = df_sorted.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1).over("asset_id").alias("_ret")
        )
        with_vols = with_ret.with_columns(
            pl.col("_ret").rolling_std(self.short_window).over("asset_id").alias("_short_vol"),
            pl.col("_ret").rolling_std(self.long_window).over("asset_id").alias("_long_vol"),
        )
        return with_vols.with_columns(
            pl.when(pl.col("_long_vol") > 0)
            .then(pl.col("_short_vol") / pl.col("_long_vol"))
            .otherwise(None)
            .alias(self.name)
        ).drop(["_ret", "_short_vol", "_long_vol"])


@dataclass(frozen=True, slots=True)
class VolumeZScoreFeature:
    """Z-score of volume relative to its rolling mean and std over
    `window` bars: (volume - rolling_mean) / rolling_std.

    Detects volume spikes (z >> 0) and dry-ups (z << 0). Null for the
    first `window - 1` rows of each asset, and wherever the rolling std
    is exactly 0 (constant volume).
    """

    name: str
    window: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.window < 2:
            raise ValueError("window must be >= 2")

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        df_sorted = df.sort(["asset_id", "timestamp"])
        with_stats = df_sorted.with_columns(
            pl.col("volume").rolling_mean(self.window).over("asset_id").alias("_vmean"),
            pl.col("volume").rolling_std(self.window).over("asset_id").alias("_vstd"),
        )
        return with_stats.with_columns(
            pl.when(pl.col("_vstd") > 0)
            .then((pl.col("volume") - pl.col("_vmean")) / pl.col("_vstd"))
            .otherwise(None)
            .alias(self.name)
        ).drop(["_vmean", "_vstd"])


@dataclass(frozen=True, slots=True)
class RelativeVolumeFeature:
    """Volume as a multiple of its rolling mean over `window` bars:
    volume / rolling_mean(volume).

    Values > 1 indicate above-average activity, < 1 below-average.
    Simpler than the z-score; useful when volume distribution is
    heavy-tailed. Null for the first `window - 1` rows of each asset,
    and wherever the rolling mean is exactly 0.
    """

    name: str
    window: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.window < 1:
            raise ValueError("window must be positive")

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        df_sorted = df.sort(["asset_id", "timestamp"])
        with_mean = df_sorted.with_columns(
            pl.col("volume").rolling_mean(self.window).over("asset_id").alias("_vmean")
        )
        return with_mean.with_columns(
            pl.when(pl.col("_vmean") > 0)
            .then(pl.col("volume") / pl.col("_vmean"))
            .otherwise(None)
            .alias(self.name)
        ).drop("_vmean")


@register_feature("ema_distance")
def _make_ema_distance(name: str, window: int) -> EMADistanceFeature:
    return EMADistanceFeature(name=name, window=window)


@register_feature("atr_percent")
def _make_atr_percent(name: str, window: int) -> ATRPercentFeature:
    return ATRPercentFeature(name=name, window=window)


@register_feature("volatility_ratio")
def _make_volatility_ratio(
    name: str, short_window: int, long_window: int
) -> VolatilityRatioFeature:
    return VolatilityRatioFeature(name=name, short_window=short_window, long_window=long_window)


@register_feature("volume_zscore")
def _make_volume_zscore(name: str, window: int) -> VolumeZScoreFeature:
    return VolumeZScoreFeature(name=name, window=window)


@register_feature("relative_volume")
def _make_relative_volume(name: str, window: int) -> RelativeVolumeFeature:
    return RelativeVolumeFeature(name=name, window=window)
