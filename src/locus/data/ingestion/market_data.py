"""
market_data.py

MarketDataIngestor orchestrates the full pipeline:

    Provider.fetch_bars → validate_ohlcv → ParquetBarStore.write

The ingestor is provider-agnostic; it receives any MarketDataProvider
and a ParquetBarStore, so the same code works for Binance, Kraken, and
CoinGecko once those providers implement the Protocol.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from locus.data.providers.base import MarketDataProvider
from locus.data.storage.parquet import ParquetBarStore
from locus.data.types import BarRequest
from locus.data.validation.ohlcv import validate_ohlcv
from locus.infrastructure.logging import LoggerFactory


@dataclass(frozen=True)
class IngestionResult:
    """Outcome of a single ingestion run."""

    rows_fetched: int
    rows_written: int
    asset_id: str
    venue: str
    timeframe: str


class MarketDataIngestor:
    """Fetch, validate, and persist OHLCV bars from a provider."""

    def __init__(
        self,
        provider: MarketDataProvider,
        store: ParquetBarStore,
    ) -> None:
        self._provider = provider
        self._store = store
        self._log = LoggerFactory.get("data.ingestion.market_data")

    def ingest(self, request: BarRequest) -> IngestionResult:
        """Run the full pipeline for one request."""
        self._log.info(
            "ingest_started",
            extra={
                "asset": request.asset_id,
                "venue": self._provider.venue,
                "timeframe": request.timeframe.value,
            },
        )

        df = self._provider.fetch_bars(request)
        if df.height == 0:
            self._log.info(
                "ingest_empty",
                extra={"asset": request.asset_id},
            )
            return IngestionResult(
                rows_fetched=0,
                rows_written=0,
                asset_id=request.asset_id,
                venue=self._provider.venue,
                timeframe=request.timeframe.value,
            )

        df = df.with_columns(pl.lit(request.asset_id).alias("asset_id"))
        df = validate_ohlcv(df)
        written = self._store.write(df)

        self._log.info(
            "ingest_completed",
            extra={
                "asset": request.asset_id,
                "rows_fetched": df.height,
                "rows_written": written,
            },
        )
        return IngestionResult(
            rows_fetched=df.height,
            rows_written=written,
            asset_id=request.asset_id,
            venue=self._provider.venue,
            timeframe=request.timeframe.value,
        )
