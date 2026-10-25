"""
funding.py

FundingIngestor orchestrates fetch → validate → write for funding
rates. Per-asset failures captured in a report; batch does not abort.

Raw layer is written BEFORE validation: if processed validation fails,
the raw batch is preserved for audit.

Author: Marco Pérez Padilla
Date:   23-10-2026
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from locus.data.providers.binance_funding import BinanceFundingProvider
from locus.data.storage.funding_store import FundingStore
from locus.data.types import ProviderError
from locus.data.universe.perp_registry import PerpUniverseRegistry
from locus.data.validation.funding import (
    check_funding_continuity,
    validate_funding,
)
from locus.infrastructure.logging import LoggerFactory


@dataclass(frozen=True, slots=True)
class FundingAssetResult:
    asset_id: str
    written: int
    error: str | None


@dataclass(frozen=True, slots=True)
class FundingIngestionReport:
    start: datetime
    end: datetime
    results: list[FundingAssetResult] = field(default_factory=list)

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
        return sum(r.written for r in self.results)


class FundingIngestor:
    def __init__(
        self,
        provider: BinanceFundingProvider,
        store: FundingStore,
        registry: PerpUniverseRegistry,
        expected_cadence: timedelta,
    ) -> None:
        self._provider = provider
        self._store = store
        self._registry = registry
        self._cadence = expected_cadence
        self._log = LoggerFactory.get("data.ingestion.funding")

    def run_batch(self, start: datetime, end: datetime) -> FundingIngestionReport:
        results: list[FundingAssetResult] = []
        for asset_id in self._registry.asset_ids():
            if not self._registry.is_listed(asset_id, "binance_futures", end):
                self._log.info(
                    "asset_not_in_universe",
                    extra={"asset": asset_id},
                )
                results.append(
                    FundingAssetResult(
                        asset_id=asset_id,
                        written=0,
                        error="not_in_universe",
                    )
                )
                continue
            try:
                df, raw_records = self._provider.fetch_funding_with_raw(asset_id, start, end)
                for rec in raw_records:
                    self._store.write_raw(
                        venue="binance_futures",
                        asset_id=asset_id,
                        symbol=rec["symbol"],
                        request_start_ms=rec["request_start_ms"],
                        request_end_ms=rec["request_end_ms"],
                        response_json=json.dumps(rec["response"]),
                    )
                if df.height == 0:
                    results.append(
                        FundingAssetResult(
                            asset_id=asset_id,
                            written=0,
                            error="empty",
                        )
                    )
                    continue
                df = validate_funding(df)
                check_funding_continuity(df, self._cadence)
                written = self._store.write(df)
                results.append(
                    FundingAssetResult(
                        asset_id=asset_id,
                        written=written,
                        error=None,
                    )
                )
            except ProviderError as exc:
                self._log.warning(
                    "funding_asset_failed",
                    extra={"asset": asset_id, "error": str(exc)},
                )
                results.append(
                    FundingAssetResult(
                        asset_id=asset_id,
                        written=0,
                        error=str(exc),
                    )
                )
        report = FundingIngestionReport(start=start, end=end, results=results)
        self._log.info(
            "funding_batch_done",
            extra={
                "assets": report.total_assets,
                "successful": report.successful,
                "failed": report.failed,
                "rows": report.total_rows,
            },
        )
        return report
