"""
test_batch.py

BatchIngestor produces a report, captures per-asset failures, and does
not abort the whole run when one asset fails.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from locus.data.ingestion.batch import BatchIngestor
from locus.data.ingestion.market_data import MarketDataIngestor
from locus.data.providers.base import BAR_SCHEMA
from locus.data.storage.parquet import ParquetBarStore
from locus.data.types import BarRequest, ProviderError, Timeframe
from locus.data.universe.types import (
    UniverseConstraint,
    UniverseSnapshot,
)
from locus.domain import Asset, MarketType
from locus.domain.types import AssetId


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


class _OkProvider:
    venue = "fake"

    def fetch_bars(self, request: BarRequest) -> pl.DataFrame:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        rows = [
            {
                "asset_id": "",
                "venue": "fake",
                "symbol": "X",
                "timeframe": "1h",
                "timestamp": base + timedelta(hours=i),
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 10.0,
            }
            for i in range(2)
        ]
        return pl.DataFrame(rows, schema=BAR_SCHEMA)


class _FailForProvider:
    venue = "fake"

    def __init__(self, fail_asset: set[str]) -> None:
        self._fail_asset = fail_asset

    def fetch_bars(self, request: BarRequest) -> pl.DataFrame:
        if request.asset_id in self._fail_asset:
            raise ProviderError("simulated failure")
        return pl.DataFrame(schema=BAR_SCHEMA)


def _snapshot(asset_ids: list[str]) -> UniverseSnapshot:
    return UniverseSnapshot(
        timestamp=_ts(),
        assets=[
            Asset(
                asset_id=AssetId(a),
                symbol=f"{a}USDT",
                market_type=MarketType.CRYPTO,
                venue="fake",
                quote_currency="USDT",
            )
            for a in asset_ids
        ],
        constraints=UniverseConstraint(),
    )


def test_batch_success(tmp_path: Path) -> None:
    store = ParquetBarStore(tmp_path)
    ingestor = MarketDataIngestor(_OkProvider(), store)
    batch = BatchIngestor(ingestor)
    report = batch.run(
        _snapshot(["BTC", "ETH"]),
        Timeframe.H1,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert report.total_assets == 2
    assert report.successful == 2
    assert report.failed == 0
    assert report.total_rows == 4


def test_batch_partial_failure(tmp_path: Path) -> None:
    store = ParquetBarStore(tmp_path)
    ingestor = MarketDataIngestor(_FailForProvider({"ETH", "SOL"}), store)
    batch = BatchIngestor(ingestor)
    report = batch.run(
        _snapshot(["BTC", "ETH", "SOL"]),
        Timeframe.H1,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert report.total_assets == 3
    assert report.successful == 1
    assert report.failed == 2
    failed_assets = {r.asset_id for r in report.results if r.error is not None}
    assert failed_assets == {"ETH", "SOL"}
