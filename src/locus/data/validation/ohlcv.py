"""
ohlcv.py

Vectorized validation of the canonical OHLCV DataFrame. Raises
ProviderResponseError on any violation. Designed to run on the whole
DataFrame at once (not per row) so it scales to millions of bars.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from locus.data.providers.base import BAR_SCHEMA
from locus.data.types import ProviderResponseError


def validate_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    """Validate schema, types, and financial invariants.

    Returns the input DataFrame unchanged if valid. Raises
    ProviderResponseError otherwise. Checks:
      - Exact schema (columns and dtypes) matches BAR_SCHEMA
      - No nulls in any column
      - All OHLC prices strictly positive
      - high >= max(open, close, low)
      - low <= min(open, close, high)
      - volume >= 0
      - timestamps strictly sorted (no duplicates)
      - timestamps timezone-aware UTC
    """
    if df.schema != BAR_SCHEMA:
        raise ProviderResponseError(f"schema mismatch: expected {BAR_SCHEMA}, got {df.schema}")

    if df.height == 0:
        return df

    nulls = df.null_count().row(0)
    if any(n > 0 for n in nulls):
        raise ProviderResponseError("null values in OHLCV data")

    if df.filter(pl.min_horizontal("open", "high", "low", "close") <= 0).height > 0:
        raise ProviderResponseError("non-positive OHLC price found")

    if df.filter(pl.col("high") < pl.max_horizontal("open", "close", "low")).height > 0:
        raise ProviderResponseError("high < max(open, close, low)")

    if df.filter(pl.col("low") > pl.min_horizontal("open", "close", "high")).height > 0:
        raise ProviderResponseError("low > min(open, close, high)")

    if df.filter(pl.col("volume") < 0).height > 0:
        raise ProviderResponseError("negative volume found")

    ts = df["timestamp"]
    if ts.is_duplicated().any():
        raise ProviderResponseError("duplicate timestamps found")

    if not ts.is_sorted():
        raise ProviderResponseError("timestamps not sorted ascending")

    return df


def check_continuity(
    df: pl.DataFrame,
    expected_bar_delta: timedelta,
    tolerance_factor: float = 1.5,
) -> None:
    """Verify each asset has no gaps larger than
    `expected_bar_delta * tolerance_factor` between consecutive bars.

    A single missing bar at a rebalance timestamp can destroy a
    backtest result: `PortfolioAccounting.equity` would value the
    position at 0 if the price is missing, producing spurious −100%
    returns. This check fails loudly before that happens.

    Per-asset. Assets may start or end at different times (that's
    normal); gaps *within* an asset's span are what we check.

    Raises `ProviderResponseError` with the offending asset and gap.
    """
    if df.height == 0:
        return
    threshold = expected_bar_delta * tolerance_factor

    for asset_id, group in df.group_by("asset_id"):
        ts = group["timestamp"].unique().sort()
        if ts.len() < 2:
            continue
        diffs = ts.diff().drop_nulls()
        max_gap = diffs.max()
        if max_gap is None:
            continue
        # `diff` on Datetime returns Duration; cast for comparison.
        gap = (
            max_gap
            if isinstance(max_gap, timedelta)
            else (pl.Series([max_gap]).cast(pl.Duration("us")).item())
        )
        if gap > threshold:
            asset_val = asset_id[0] if isinstance(asset_id, tuple) else asset_id
            raise ProviderResponseError(
                f"continuity gap in {asset_val!r}: "
                f"max gap {gap} > threshold {threshold} "
                f"(expected bar delta {expected_bar_delta})"
            )
