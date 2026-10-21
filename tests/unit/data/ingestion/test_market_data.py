"""
test_market_data.py

MarketDataIngestor end-to-end with a fake provider (no network).

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from locus.data.ingestion.market_data import MarketDataIngestor
from locus.data.providers.base import BAR_SCHEMA
from locus.data.storage.parquet import ParquetBarStore
from locus.data.types import BarRequest, Timeframe


class _FakeProvider:
    venue = "fake"

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df
        self.last_request: BarRequest | None = None

    def fetch_bars(self, request: BarRequest) -> pl.DataFrame:
        self.last_request = request
        return self._df


def _df(n: int = 2) -> pl.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(n):
        rows.append(
            {
                "asset_id": "",
                "venue": "fake",
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


def _request() -> BarRequest:
    return BarRequest(
        asset_id="BTC",
        timeframe=Timeframe.H1,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
    )


def test_ingest_overwrites_asset_id_and_writes(tmp_path: Path) -> None:
    store = ParquetBarStore(tmp_path)
    provider = _FakeProvider(_df(2))
    ingestor = MarketDataIngestor(provider, store)
    result = ingestor.ingest(_request())
    assert result.rows_fetched == 2
    assert result.rows_written == 2
    assert result.asset_id == "BTC"
    read_back = store.read(
        venue="fake",
        asset_id="BTC",
        timeframe=Timeframe.H1,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert read_back.height == 2
    assert read_back["asset_id"].unique().to_list() == ["BTC"]


def test_ingest_empty_provider_returns_zero(tmp_path: Path) -> None:
    store = ParquetBarStore(tmp_path)
    provider = _FakeProvider(pl.DataFrame(schema=BAR_SCHEMA))
    ingestor = MarketDataIngestor(provider, store)
    result = ingestor.ingest(_request())
    assert result.rows_fetched == 0
    assert result.rows_written == 0
