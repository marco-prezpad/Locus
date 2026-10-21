"""
fx.py

Batch and incremental ingestion for FX rates. Each pair has a fallback
chain; the ingestor tries the chain, writes what comes back, and
reports per-pair outcomes.

`run_batch`: full historical download. Ignores the cache; the caller
explicitly asks for a full refresh.
`run_incremental`: fetches only the missing recent days. Skips pairs
that already have yesterday's rate in the store, from any source.

No scheduler is built in. Both methods are invoked manually or by an
external cron (OS scheduler or GitHub Actions). See docs/currency.md.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from locus.data.fx.fallback_fx import FallbackFxProvider
from locus.data.fx.store import FxRateStore
from locus.data.fx.types import FxProviderError
from locus.infrastructure.logging import LoggerFactory


@dataclass(frozen=True, slots=True)
class FxPairIngestionResult:
    """Outcome for one pair inside an ingestion run."""

    pair: str
    source: str | None
    written: int
    skipped: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class FxIngestionReport:
    """Summary of a full FX ingestion run."""

    start: datetime
    end: datetime
    results: list[FxPairIngestionResult] = field(default_factory=list)

    @property
    def total_pairs(self) -> int:
        return len(self.results)

    @property
    def written_pairs(self) -> int:
        return sum(1 for r in self.results if not r.skipped and r.reason is None)

    @property
    def skipped_pairs(self) -> int:
        return sum(1 for r in self.results if r.skipped)

    @property
    def failed_pairs(self) -> int:
        return sum(1 for r in self.results if not r.skipped and r.reason is not None)

    @property
    def total_rows(self) -> int:
        return sum(r.written for r in self.results)


class FxRateIngestor:
    """Fill the FX rate store from fallback chains per pair."""

    def __init__(
        self,
        chains: dict[str, FallbackFxProvider],
        store: FxRateStore,
    ) -> None:
        if not chains:
            raise ValueError("chains must be non-empty")
        self._chains = chains
        self._store = store
        self._log = LoggerFactory.get("data.ingestion.fx")

    def run_batch(self, start: datetime, end: datetime) -> FxIngestionReport:
        """Download the full [start, end) range for every pair. The
        cache is ignored; the caller explicitly asks for a full
        refresh."""
        self._log.info(
            "fx_batch_started",
            extra={
                "pairs": list(self._chains),
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )
        results: list[FxPairIngestionResult] = []
        for pair, chain in self._chains.items():
            results.append(self._ingest_pair(pair, chain, start, end))
        report = FxIngestionReport(start=start, end=end, results=results)
        self._log_summary("fx_batch_completed", report)
        return report

    def run_incremental(self, lookback_days: int = 7) -> FxIngestionReport:
        """Fetch the last `lookback_days` days for every pair, skipping
        pairs that already have yesterday's rate in the store (from any
        source)."""
        end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
            days=1
        )
        start = end - timedelta(days=lookback_days)

        self._log.info(
            "fx_incremental_started",
            extra={
                "pairs": list(self._chains),
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )
        results: list[FxPairIngestionResult] = []
        for pair, chain in self._chains.items():
            if self._has_recent(pair, end):
                self._log.info(
                    "fx_pair_up_to_date",
                    extra={"pair": pair},
                )
                results.append(
                    FxPairIngestionResult(
                        pair=pair,
                        source=None,
                        written=0,
                        skipped=True,
                        reason="up_to_date",
                    )
                )
                continue
            results.append(self._ingest_pair(pair, chain, start, end))

        report = FxIngestionReport(start=start, end=end, results=results)
        self._log_summary("fx_incremental_completed", report)
        return report

    def _has_recent(self, pair: str, end: datetime) -> bool:
        """True if any source has a rate at or after `end - 1 day`."""
        window_start = end - timedelta(days=1)
        for source in self._store.available_sources(pair):
            df = self._store.read(pair, source, window_start, end)
            if df.height > 0:
                return True
        return False

    def _ingest_pair(
        self,
        pair: str,
        chain: FallbackFxProvider,
        start: datetime,
        end: datetime,
    ) -> FxPairIngestionResult:
        try:
            df = chain.fetch_rates(pair, start, end)
        except FxProviderError as exc:
            self._log.warning(
                "fx_pair_failed",
                extra={"pair": pair, "error": str(exc)},
            )
            return FxPairIngestionResult(
                pair=pair,
                source=None,
                written=0,
                skipped=False,
                reason=str(exc),
            )

        if df.height == 0:
            return FxPairIngestionResult(
                pair=pair,
                source=None,
                written=0,
                skipped=False,
                reason="empty",
            )

        written = self._store.write(df, source=chain.source)
        return FxPairIngestionResult(
            pair=pair,
            source=chain.source,
            written=written,
            skipped=False,
        )

    def _log_summary(self, event: str, report: FxIngestionReport) -> None:
        self._log.info(
            event,
            extra={
                "total_pairs": report.total_pairs,
                "written_pairs": report.written_pairs,
                "skipped_pairs": report.skipped_pairs,
                "failed_pairs": report.failed_pairs,
                "total_rows": report.total_rows,
            },
        )
