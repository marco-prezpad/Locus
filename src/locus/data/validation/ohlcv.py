"""
ohlcv.py

Vectorized validation of the canonical OHLCV DataFrame. Raises
ProviderResponseError on any violation. Designed to run on the whole
DataFrame at once (not per row) so it scales to millions of bars.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

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
