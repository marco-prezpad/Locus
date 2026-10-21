"""
volatility.py

Volatility features: rolling standard deviation of one-period returns
per asset. Captures recent realized volatility.

Two-pass: compute 1-period returns, then rolling std. `annualization`
is NOT applied here; features are in raw per-bar units so they remain
comparable across timeframes. Downstream models can annualize.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from locus.features.base import register_feature


@dataclass(frozen=True, slots=True)
class RollingVolatilityFeature:
    """Rolling standard deviation of 1-period returns over `window`
    bars. Null for the first `window` rows of each asset (one for the
    return, window-1 for the std).
    """

    name: str
    window: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.window < 2:
            raise ValueError("window must be >= 2 for std")

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        df_sorted = df.sort(["asset_id", "timestamp"])
        with_ret = df_sorted.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1).over("asset_id").alias("_ret")
        )
        return with_ret.with_columns(
            pl.col("_ret").rolling_std(self.window).over("asset_id").alias(self.name)
        ).drop("_ret")


@register_feature("rolling_volatility")
def _make_rolling_volatility(name: str, window: int) -> RollingVolatilityFeature:
    return RollingVolatilityFeature(name=name, window=window)
