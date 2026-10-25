"""
funding.py

Validation of funding DataFrames against FUNDING_SCHEMA and continuity
of funding events per asset.

Continuity: funding cadence is 8h for BTC/ETH/SOL/BNB/XRP. A gap > 1.5×
the expected cadence (>12h) raises ProviderResponseError.

Author: Marco Pérez Padilla
Date:   23-10-2026
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from locus.data.providers.funding_base import FUNDING_SCHEMA
from locus.data.types import ProviderResponseError


def validate_funding(df: pl.DataFrame) -> pl.DataFrame:
    """Validate schema, no nulls, finite rates, sorted and deduped."""
    if df.schema != FUNDING_SCHEMA:
        raise ProviderResponseError(
            f"funding schema mismatch: expected {FUNDING_SCHEMA}, got {df.schema}"
        )
    if df.height == 0:
        return df

    nulls = df.null_count().row(0)
    if any(n > 0 for n in nulls):
        raise ProviderResponseError("null values in funding data")

    if df.filter(~pl.col("funding_rate").is_finite()).height > 0:
        raise ProviderResponseError("non-finite funding_rate found")

    ts = df["timestamp"]
    if ts.is_duplicated().any():
        raise ProviderResponseError("duplicate funding timestamps")
    if not ts.is_sorted():
        raise ProviderResponseError("funding timestamps not sorted")

    return df


def check_funding_continuity(
    df: pl.DataFrame,
    expected_cadence: timedelta,
    tolerance_factor: float = 1.5,
) -> None:
    """Verify no gap > tolerance_factor × expected_cadence within each
    asset's series."""
    if df.height == 0:
        return
    threshold = expected_cadence * tolerance_factor
    for asset_id, group in df.group_by("asset_id"):
        ts = group["timestamp"].unique().sort()
        if ts.len() < 2:
            continue
        diffs = ts.diff().drop_nulls()
        max_gap = diffs.max()
        if max_gap is None:
            continue
        gap = (
            max_gap
            if isinstance(max_gap, timedelta)
            else (pl.Series([max_gap]).cast(pl.Duration("us")).item())
        )
        if gap > threshold:
            asset_val = asset_id[0] if isinstance(asset_id, tuple) else asset_id
            raise ProviderResponseError(
                f"funding continuity gap in {asset_val!r}: "
                f"max gap {gap} > threshold {threshold} "
                f"(expected cadence {expected_cadence})"
            )
