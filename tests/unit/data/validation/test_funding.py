"""
test_funding.py

Funding validation and continuity.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.data.providers.funding_base import FUNDING_SCHEMA
from locus.data.types import ProviderResponseError
from locus.data.validation.funding import (
    check_funding_continuity,
    validate_funding,
)


def _df(n: int = 5) -> pl.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        {
            "asset_id": "BTC",
            "venue": "binance_futures",
            "symbol": "BTCUSDT",
            "timestamp": base + timedelta(hours=8 * i),
            "funding_rate": 0.0001,
        }
        for i in range(n)
    ]
    return pl.DataFrame(rows, schema=FUNDING_SCHEMA)


def test_valid_funding_passes() -> None:
    assert validate_funding(_df()).height == 5


def test_schema_mismatch_raises() -> None:
    df = _df().drop("funding_rate")
    with pytest.raises(ProviderResponseError):
        validate_funding(df)


def test_duplicate_timestamps_raise() -> None:
    df = _df(3).with_columns(pl.lit(datetime(2026, 1, 1, tzinfo=UTC)).alias("timestamp"))
    with pytest.raises(ProviderResponseError):
        validate_funding(df)


def test_continuity_passes() -> None:
    check_funding_continuity(_df(5), timedelta(hours=8))


def test_continuity_detects_gap() -> None:
    df = _df(5).with_columns(
        pl.when(pl.int_range(pl.len()) == 2)
        .then(pl.col("timestamp") + timedelta(hours=24))
        .otherwise(pl.col("timestamp"))
        .alias("timestamp")
    )
    with pytest.raises(ProviderResponseError, match="continuity gap"):
        check_funding_continuity(df, timedelta(hours=8))
