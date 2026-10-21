"""
test_ohlcv.py

Vectorized OHLCV validation: schema, nulls, OHLC invariants, ordering.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.data.providers.base import BAR_SCHEMA
from locus.data.types import ProviderResponseError
from locus.data.validation.ohlcv import validate_ohlcv


def _valid_df(n: int = 3) -> pl.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(n):
        rows.append(
            {
                "asset_id": "BTC",
                "venue": "binance",
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "timestamp": base + timedelta(hours=i),
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 10.0,
            }
        )
    return pl.DataFrame(rows, schema=BAR_SCHEMA)


def test_valid_df_passes() -> None:
    df = _valid_df()
    assert validate_ohlcv(df).height == 3


def test_empty_df_passes() -> None:
    assert validate_ohlcv(pl.DataFrame(schema=BAR_SCHEMA)).height == 0


def test_schema_mismatch_raises() -> None:
    df = _valid_df().drop("volume")
    with pytest.raises(ProviderResponseError):
        validate_ohlcv(df)


def test_null_value_raises() -> None:
    df = _valid_df().with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("open")).alias("open")
    )
    with pytest.raises(ProviderResponseError):
        validate_ohlcv(df)


def test_non_positive_price_raises() -> None:
    df = _valid_df().with_columns(pl.lit(0.0).alias("open"))
    with pytest.raises(ProviderResponseError):
        validate_ohlcv(df)


def test_high_below_close_raises() -> None:
    df = _valid_df().with_columns(pl.lit(50.0).alias("high"))
    with pytest.raises(ProviderResponseError):
        validate_ohlcv(df)


def test_negative_volume_raises() -> None:
    df = _valid_df().with_columns(pl.lit(-1.0).alias("volume"))
    with pytest.raises(ProviderResponseError):
        validate_ohlcv(df)


def test_duplicate_timestamps_raises() -> None:
    df = _valid_df(2).with_columns(pl.lit(datetime(2026, 1, 1, tzinfo=UTC)).alias("timestamp"))
    with pytest.raises(ProviderResponseError):
        validate_ohlcv(df)


def test_unsorted_timestamps_raises() -> None:
    df = _valid_df(3).reverse()
    with pytest.raises(ProviderResponseError):
        validate_ohlcv(df)
