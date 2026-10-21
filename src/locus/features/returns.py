"""
returns.py

Return features: simple returns and log returns over n periods, per
asset. `ReturnFeature(n).name = "return_{n}"` unless overridden.

`return_n` at t = close[t] / close[t-n] - 1, computed per asset.
Null for the first n rows of each asset (insufficient history).

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from locus.features.base import register_feature


@dataclass(frozen=True, slots=True)
class ReturnFeature:
    """Simple return over n periods: close[t]/close[t-n] - 1."""

    name: str
    n: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.n <= 0:
            raise ValueError("n must be positive")

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.sort(["asset_id", "timestamp"]).with_columns(
            (pl.col("close") / pl.col("close").shift(self.n) - 1).over("asset_id").alias(self.name)
        )


@dataclass(frozen=True, slots=True)
class LogReturnFeature:
    """Log return over n periods: ln(close[t]/close[t-n])."""

    name: str
    n: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.n <= 0:
            raise ValueError("n must be positive")

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.sort(["asset_id", "timestamp"]).with_columns(
            (pl.col("close") / pl.col("close").shift(self.n))
            .log()
            .over("asset_id")
            .alias(self.name)
        )


@register_feature("return")
def _make_return(name: str, n: int) -> ReturnFeature:
    return ReturnFeature(name=name, n=n)


@register_feature("log_return")
def _make_log_return(name: str, n: int) -> LogReturnFeature:
    return LogReturnFeature(name=name, n=n)
