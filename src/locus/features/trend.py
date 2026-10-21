"""
trend.py

Trend features: distance of close to a moving average. Positive values
mean close is above the MA (uptrend); negative means below.

Two-pass implementation: compute the MA per asset, then the normalized
distance. Avoids nested .over() semantics, which polars handles
correctly but is harder to reason about.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from locus.features.base import register_feature


@dataclass(frozen=True, slots=True)
class MovingAverageDistanceFeature:
    """Distance from close to a simple moving average of `window` bars:
    (close - MA) / MA. Null for the first `window - 1` rows of each asset.
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
        with_ma = df_sorted.with_columns(
            pl.col("close").rolling_mean(self.window).over("asset_id").alias("_ma")
        )
        return with_ma.with_columns(
            ((pl.col("close") - pl.col("_ma")) / pl.col("_ma")).alias(self.name)
        ).drop("_ma")


@register_feature("ma_distance")
def _make_ma_distance(name: str, window: int) -> MovingAverageDistanceFeature:
    return MovingAverageDistanceFeature(name=name, window=window)
