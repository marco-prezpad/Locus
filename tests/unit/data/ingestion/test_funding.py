"""
test_funding.py (ingestion)

FundingIngestor batch behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from locus.data.ingestion.funding import FundingIngestor
from locus.data.providers.funding_base import FUNDING_SCHEMA
from locus.data.storage.funding_store import FundingStore
from locus.data.types import ProviderError
from locus.data.universe.perp_registry import PerpUniverseRegistry
from locus.data.universe.perp_types import PerpListing


def _registry() -> PerpUniverseRegistry:
    return PerpUniverseRegistry(
        [
            PerpListing(
                asset_id="BTC",
                venue="binance_futures",
                list_date=datetime(2020, 1, 1, tzinfo=UTC),
            ),
            PerpListing(
                asset_id="ETH",
                venue="binance_futures",
                list_date=datetime(2020, 1, 1, tzinfo=UTC),
            ),
        ]
    )


class _OkProvider:
    venue = "binance_futures"

    def fetch_funding_with_raw(self, asset_id, start, end):
        base = datetime(2026, 1, 1, tzinfo=UTC)
        rows = [
            {
                "asset_id": asset_id,
                "venue": self.venue,
                "symbol": f"{asset_id}USDT",
                "timestamp": base + timedelta(hours=8 * i),
                "funding_rate": 0.0001,
            }
            for i in range(3)
        ]
        df = pl.DataFrame(rows, schema=FUNDING_SCHEMA)
        raw = [
            {
                "symbol": f"{asset_id}USDT",
                "request_start_ms": 0,
                "request_end_ms": 0,
                "response": [],
            }
        ]
        return df, raw


class _FailProvider:
    venue = "binance_futures"

    def fetch_funding_with_raw(self, asset_id, start, end):
        raise ProviderError("boom")


def _run(provider, tmp_path: Path):
    store = FundingStore(tmp_path / "processed", tmp_path / "raw")
    ingestor = FundingIngestor(
        provider=provider,  # type: ignore[arg-type]
        store=store,
        registry=_registry(),
        expected_cadence=timedelta(hours=8),
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 5, tzinfo=UTC)
    return ingestor.run_batch(start, end)


def test_batch_success(tmp_path: Path) -> None:
    report = _run(_OkProvider(), tmp_path)
    assert report.successful == 2
    assert report.failed == 0
    assert report.total_rows == 6


def test_batch_failure_captured(tmp_path: Path) -> None:
    report = _run(_FailProvider(), tmp_path)
    assert report.successful == 0
    assert report.failed == 2
    assert report.results[0].error == "boom"
