"""
test_parquet.py

ParquetBarStore round-trip: write then read partitions by year.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from locus.data.providers.base import BAR_SCHEMA
from locus.data.storage.parquet import ParquetBarStore
from locus.data.types import ProviderResponseError, Timeframe


def _df_spanning_years() -> pl.DataFrame:
    rows = []
    for year, hour in [(2024, 23), (2025, 0), (2025, 1)]:
        rows.append(
            {
                "asset_id": "BTC",
                "venue": "binance",
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "timestamp": datetime(year, 12, 31, hour, tzinfo=UTC)
                if year == 2024
                else datetime(year, 1, 1, hour, tzinfo=UTC),
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 10.0,
            }
        )
    return pl.DataFrame(rows, schema=BAR_SCHEMA).sort("timestamp")


def test_write_creates_yearly_partitions(tmp_path: Path) -> None:
    store = ParquetBarStore(tmp_path)
    df = _df_spanning_years()
    written = store.write(df)
    assert written == 3
    assert (
        tmp_path
        / "market=crypto"
        / "venue=binance"
        / "asset=BTC"
        / "timeframe=1h"
        / "year=2024"
        / "bars.parquet"
    ).exists()
    assert (
        tmp_path
        / "market=crypto"
        / "venue=binance"
        / "asset=BTC"
        / "timeframe=1h"
        / "year=2025"
        / "bars.parquet"
    ).exists()


def test_read_returns_requested_range(tmp_path: Path) -> None:
    store = ParquetBarStore(tmp_path)
    store.write(_df_spanning_years())
    df = store.read(
        venue="binance",
        asset_id="BTC",
        timeframe=Timeframe.H1,
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 2, tzinfo=UTC),
    )
    assert df.height == 2


def test_read_empty_when_no_files(tmp_path: Path) -> None:
    store = ParquetBarStore(tmp_path)
    df = store.read(
        venue="binance",
        asset_id="BTC",
        timeframe=Timeframe.H1,
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 2, tzinfo=UTC),
    )
    assert df.height == 0


def test_write_multiple_assets_raises(tmp_path: Path) -> None:
    store = ParquetBarStore(tmp_path)
    df = _df_spanning_years().with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit("ETH"))
        .otherwise(pl.col("asset_id"))
        .alias("asset_id")
    )
    with pytest.raises(ProviderResponseError):
        store.write(df)
