"""
batch.py

BatchIngestor downloads OHLCV bars for an entire universe snapshot.
It iterates over the snapshot assets, calls MarketDataIngestor for each,
and produces a DownloadReport. Failures per asset are captured in the
report, not raised: a universe with 50 assets should not abort because
one is delisted or missing.

Rate limiting: a small sleep between assets keeps the batch well within
the provider's public rate limit. Per-request retries are handled by the
provider itself (tenacity).

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from locus.data.ingestion.market_data import MarketDataIngestor
from locus.data.types import BarRequest, ProviderError, Timeframe
from locus.data.universe.types import UniverseSnapshot
from locus.infrastructure.logging import LoggerFactory


@dataclass(frozen=True, slots=True)
class AssetDownloadResult:
    """Outcome of one asset download inside a batch run."""

    asset_id: str
    venue: str
    rows_written: int
    error: str | None


@dataclass(frozen=True, slots=True)
class DownloadReport:
    """Summary of a full batch run over a universe snapshot."""

    timestamp: datetime
    timeframe: Timeframe
    start: datetime
    end: datetime
    results: list[AssetDownloadResult] = field(default_factory=list)

    @property
    def total_assets(self) -> int:
        return len(self.results)

    @property
    def successful(self) -> int:
        return sum(1 for r in self.results if r.error is None)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.error is not None)

    @property
    def total_rows(self) -> int:
        return sum(r.rows_written for r in self.results)


class BatchIngestor:
    """Download bars for every asset in a universe snapshot."""

    def __init__(
        self,
        ingestor: MarketDataIngestor,
        sleep_between_assets_seconds: float = 0.0,
    ) -> None:
        self._ingestor = ingestor
        self._sleep = sleep_between_assets_seconds
        self._log = LoggerFactory.get("data.ingestion.batch")

    def run(
        self,
        snapshot: UniverseSnapshot,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> DownloadReport:
        """Iterate the snapshot, download each asset, return a report."""
        self._log.info(
            "batch_started",
            extra={
                "assets": len(snapshot.assets),
                "timeframe": timeframe.value,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )

        results: list[AssetDownloadResult] = []
        for asset in snapshot.assets:
            try:
                res = self._ingestor.ingest(
                    BarRequest(
                        asset_id=asset.asset_id,
                        timeframe=timeframe,
                        start=start,
                        end=end,
                    )
                )
                results.append(
                    AssetDownloadResult(
                        asset_id=asset.asset_id,
                        venue=asset.venue,
                        rows_written=res.rows_written,
                        error=None,
                    )
                )
            except ProviderError as exc:
                self._log.warning(
                    "asset_failed",
                    extra={
                        "asset": asset.asset_id,
                        "venue": asset.venue,
                        "error": str(exc),
                    },
                )
                results.append(
                    AssetDownloadResult(
                        asset_id=asset.asset_id,
                        venue=asset.venue,
                        rows_written=0,
                        error=str(exc),
                    )
                )
            if self._sleep > 0:
                time.sleep(self._sleep)

        report = DownloadReport(
            timestamp=snapshot.timestamp,
            timeframe=timeframe,
            start=start,
            end=end,
            results=results,
        )
        self._log.info(
            "batch_completed",
            extra={
                "assets": report.total_assets,
                "successful": report.successful,
                "failed": report.failed,
                "rows": report.total_rows,
            },
        )
        return report
