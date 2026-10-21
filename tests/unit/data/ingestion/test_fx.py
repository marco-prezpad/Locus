"""
test_fx.py

FxRateIngestor: batch, incremental with cache skip, per-pair failure
capture.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from locus.data.fx.base import FX_PROVIDER_SCHEMA
from locus.data.fx.fallback_fx import FallbackFxProvider
from locus.data.fx.store import FxRateStore
from locus.data.fx.types import FxProviderError
from locus.data.ingestion.fx import FxRateIngestor


def _df_for_range(start: datetime, end: datetime, rate: float = 1.10) -> pl.DataFrame:
    days = (end - start).days
    rows = [
        {
            "pair": "EURUSD",
            "date": start + timedelta(days=i),
            "rate": rate,
        }
        for i in range(days)
    ]
    return pl.DataFrame(rows, schema=FX_PROVIDER_SCHEMA)


class _OkProvider:
    source = "frankfurter"

    def __init__(self) -> None:
        self.calls: list[tuple[datetime, datetime]] = []

    def fetch_rates(self, pair, start, end):
        self.calls.append((start, end))
        return _df_for_range(start, end)


class _FailProvider:
    source = "fail"

    def fetch_rates(self, pair, start, end):
        raise FxProviderError("simulated")


def test_run_batch_writes_all_pairs(tmp_path: Path) -> None:
    store = FxRateStore(tmp_path)
    provider = _OkProvider()
    ingestor = FxRateIngestor(
        chains={"EURUSD": FallbackFxProvider([provider])},
        store=store,
    )
    report = ingestor.run_batch(
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 5, tzinfo=UTC),
    )
    assert report.written_pairs == 1
    assert report.total_rows == 4


def test_run_incremental_skips_when_up_to_date(tmp_path: Path) -> None:
    store = FxRateStore(tmp_path)
    # Seed with today's rate (00:00 UTC). _has_recent considers the
    # pair up to date if any source has a rate in [today, tomorrow).
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    store.write(
        pl.DataFrame(
            [{"pair": "EURUSD", "date": today, "rate": 1.10}],
            schema=FX_PROVIDER_SCHEMA,
        ),
        "frankfurter",
    )
    provider = _OkProvider()
    ingestor = FxRateIngestor(
        chains={"EURUSD": FallbackFxProvider([provider])},
        store=store,
    )
    report = ingestor.run_incremental(lookback_days=7)
    assert report.skipped_pairs == 1
    assert provider.calls == []


def test_run_incremental_fetches_when_stale(tmp_path: Path) -> None:
    store = FxRateStore(tmp_path)
    provider = _OkProvider()
    ingestor = FxRateIngestor(
        chains={"EURUSD": FallbackFxProvider([provider])},
        store=store,
    )
    report = ingestor.run_incremental(lookback_days=7)
    assert report.written_pairs == 1
    assert len(provider.calls) == 1


def test_run_incremental_captures_failure(tmp_path: Path) -> None:
    store = FxRateStore(tmp_path)
    ingestor = FxRateIngestor(
        chains={"EURUSD": FallbackFxProvider([_FailProvider()])},
        store=store,
    )
    report = ingestor.run_incremental(lookback_days=7)
    assert report.failed_pairs == 1
    assert "simulated" in (report.results[0].reason or "")
