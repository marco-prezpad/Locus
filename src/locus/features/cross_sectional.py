"""
cross_sectional.py

Cross-sectional features: for each timestamp, rank or normalize an
existing per-asset feature across all assets. The source column must
exist in the DataFrame; pipeline ordering enforces this.

All cross-sectional aggregations use `.over("timestamp")`, so they see
only data at the current timestamp (never future). Cross-sectional
leakage tests verify this with multi-asset panels.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from locus.features.base import register_feature


def _require_source(df: pl.DataFrame, name: str, source: str) -> None:
    """Raise if `source` is not present. Cross-sectional features need
    their source column to be computed earlier in the pipeline."""
    if source not in df.columns:
        raise ValueError(
            f"cross-sectional feature {name!r} requires column {source!r}, "
            f"but it is not present in the DataFrame. Ensure {source!r} is "
            f"computed earlier in the pipeline (typically a per-asset "
            f"feature)."
        )


@dataclass(frozen=True, slots=True)
class CrossSectionalPercentileRankFeature:
    """Percentile rank of `source` across assets at each timestamp, in
    (0, 1), using the midpoint formula `(rank - 0.5) / n_non_null`.

    Higher source value → higher percentile. For "lower is better"
    sources (e.g. volatility), negate the source or use a z-score
    instead.

    Null where `source` is null. Values close to 0 mean "worst in the
    cross-section at this timestamp"; close to 1 mean "best".
    """

    name: str
    source: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.source:
            raise ValueError("source must be non-empty")
        if self.source == self.name:
            raise ValueError("name must differ from source")

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        _require_source(df, self.name, self.source)
        non_null_count = pl.col(self.source).count().over("timestamp")
        return df.with_columns(
            pl.when(pl.col(self.source).is_not_null())
            .then(
                (pl.col(self.source).rank(method="average").over("timestamp") - 0.5)
                / non_null_count
            )
            .otherwise(None)
            .alias(self.name)
        )


@dataclass(frozen=True, slots=True)
class CrossSectionalZScoreFeature:
    """Z-score of `source` across assets at each timestamp:
    (source - cross_sectional_mean) / cross_sectional_std.

    Positive means the asset is above the cross-sectional mean at that
    timestamp. Null where `source` is null, where the group has fewer
    than 2 non-null values, or where std is 0 (all tied).
    """

    name: str
    source: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.source:
            raise ValueError("source must be non-empty")
        if self.source == self.name:
            raise ValueError("name must differ from source")

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        _require_source(df, self.name, self.source)
        mean_col = pl.col(self.source).mean().over("timestamp")
        std_col = pl.col(self.source).std().over("timestamp")
        return df.with_columns(
            pl.when(pl.col(self.source).is_not_null() & (std_col > 0))
            .then((pl.col(self.source) - mean_col) / std_col)
            .otherwise(None)
            .alias(self.name)
        )


@dataclass(frozen=True, slots=True)
class RelativeMomentumVsMarketFeature:
    """Asset's `source` value minus the equal-weighted cross-sectional
    mean of `source` at the same timestamp.

    For `source=return_20`, this is the asset's 20-bar momentum minus
    the equal-weighted market momentum. Positive = outperforming the
    market over the window; negative = underperforming.

    Null where `source` is null. The cross-sectional mean itself is
    computed over non-null sources at that timestamp.
    """

    name: str
    source: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.source:
            raise ValueError("source must be non-empty")
        if self.source == self.name:
            raise ValueError("name must differ from source")

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        _require_source(df, self.name, self.source)
        market_mean = pl.col(self.source).mean().over("timestamp")
        return df.with_columns(
            pl.when(pl.col(self.source).is_not_null())
            .then(pl.col(self.source) - market_mean)
            .otherwise(None)
            .alias(self.name)
        )


@register_feature("xs_percentile_rank")
def _make_xs_percentile(name: str, source: str) -> CrossSectionalPercentileRankFeature:
    return CrossSectionalPercentileRankFeature(name=name, source=source)


@register_feature("xs_zscore")
def _make_xs_zscore(name: str, source: str) -> CrossSectionalZScoreFeature:
    return CrossSectionalZScoreFeature(name=name, source=source)


@register_feature("relative_momentum")
def _make_relative_momentum(name: str, source: str) -> RelativeMomentumVsMarketFeature:
    return RelativeMomentumVsMarketFeature(name=name, source=source)
