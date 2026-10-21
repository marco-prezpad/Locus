"""
converter.py

Convert amounts between currencies using the FX rates in a store. The
converter routes through a small currency graph built from the available
pairs; for the MVP (EURUSD, USDTUSD) the graph is a star centered on
USD, so any pair of currencies is at most 2 hops away.

Weekend and holiday handling: if the requested date has no rate, the
converter falls back to the most recent rate within
`stale_days_threshold`. Older rates raise FxRateNotFoundError.

The converter never invents rates: if no path exists between the two
currencies, or if no source has a fresh enough rate, it fails loudly.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from locus.data.fx.store import FxRateStore
from locus.data.fx.types import FxPair, FxRateNotFoundError
from locus.infrastructure.logging import LoggerFactory


@dataclass(frozen=True, slots=True)
class ConversionPathStep:
    """One step of a conversion path: a pair and how to apply its rate."""

    pair: str
    direction: str  # "multiply" if we go base->quote, "divide" if quote->base


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """Detailed conversion outcome for auditing.

    `amount_out` is the final amount. `rate` is the cumulative exchange
    rate from `from_currency` to `to_currency`. `path` is the list of
    pairs used. `date_used` is the actual date of the rates used (may be
    earlier than the requested date for weekends/holidays). `sources` is
    the source per pair.
    """

    amount_in: float
    from_currency: str
    to_currency: str
    amount_out: float
    rate: float
    path: list[ConversionPathStep]
    date_requested: datetime
    date_used: datetime
    sources: dict[str, str]


class FxConverter:
    """Convert amounts between currencies via the rates in a store."""

    def __init__(
        self,
        store: FxRateStore,
        stale_days_threshold: int = 7,
        source_preferences: dict[str, list[str]] | None = None,
    ) -> None:
        self._store = store
        self._stale_days = stale_days_threshold
        self._source_prefs = source_preferences or {}
        self._log = LoggerFactory.get("data.fx.converter")

    def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        date: datetime,
    ) -> float:
        """Convert `amount` from `from_currency` to `to_currency` at
        `date`. Same-currency conversions return `amount` unchanged."""
        result = self.convert_with_details(amount, from_currency, to_currency, date)
        return result.amount_out

    def convert_with_details(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        date: datetime,
    ) -> ConversionResult:
        """Like `convert`, but returns the full path, rate, and sources
        used. Useful for audit trails in the final evaluation."""
        if date.tzinfo is None:
            raise ValueError("date must be timezone-aware")
        from_c = from_currency.upper()
        to_c = to_currency.upper()

        if from_c == to_c:
            return ConversionResult(
                amount_in=amount,
                from_currency=from_c,
                to_currency=to_c,
                amount_out=amount,
                rate=1.0,
                path=[],
                date_requested=date,
                date_used=date,
                sources={},
            )

        path = self._route(from_c, to_c)
        rate, date_used, sources = self._compute_path_rate(path, date)
        return ConversionResult(
            amount_in=amount,
            from_currency=from_c,
            to_currency=to_c,
            amount_out=amount * rate,
            rate=rate,
            path=path,
            date_requested=date,
            date_used=date_used,
            sources=sources,
        )

    def _route(self, from_c: str, to_c: str) -> list[ConversionPathStep]:
        """Find the shortest path between two currencies in the pair
        graph. Raises FxRateNotFoundError if no path exists."""
        graph = self._build_graph()
        if from_c not in graph:
            raise FxRateNotFoundError(f"no FX rates available for currency {from_c!r}")

        queue: deque[tuple[str, list[ConversionPathStep]]] = deque([(from_c, [])])
        visited: set[str] = {from_c}
        while queue:
            currency, path = queue.popleft()
            if currency == to_c:
                return path
            for neighbor, pair, direction in graph[currency]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(
                    (
                        neighbor,
                        path + [ConversionPathStep(pair=pair, direction=direction)],
                    )
                )

        raise FxRateNotFoundError(f"no conversion path from {from_c!r} to {to_c!r}")

    def _build_graph(
        self,
    ) -> dict[str, list[tuple[str, str, str]]]:
        """Adjacency list of the currency graph.

        Each edge is (neighbor_currency, pair_name, direction). The
        graph is undirected in the sense that both directions of each
        pair are added.
        """
        graph: dict[str, list[tuple[str, str, str]]] = {}
        for pair in self._store.available_pairs():
            p = FxPair.parse(pair)
            graph.setdefault(p.base, []).append((p.quote, pair, "multiply"))
            graph.setdefault(p.quote, []).append((p.base, pair, "divide"))
        return graph

    def _compute_path_rate(
        self, path: list[ConversionPathStep], date: datetime
    ) -> tuple[float, datetime, dict[str, str]]:
        """Compute the cumulative rate along a path. All steps must use
        the same `date_used` (the earliest fallback date across steps)
        so the conversion is internally consistent."""
        if not path:
            return 1.0, date, {}

        rates: list[tuple[float, str, datetime]] = []
        for step in path:
            rate, step_date, source = self._lookup_rate(step.pair, date)
            rates.append((rate, source, step_date))

        date_used = min(r[2] for r in rates)
        cumulative = 1.0
        sources: dict[str, str] = {}
        for (rate, source, _), step in zip(rates, path, strict=False):
            if step.direction == "multiply":
                cumulative *= rate
            else:
                cumulative /= rate
            sources[step.pair] = source
        return cumulative, date_used, sources

    def _lookup_rate(self, pair: str, date: datetime) -> tuple[float, datetime, str]:
        """Find the most recent rate for `pair` at or before `date`,
        within the staleness threshold. Returns (rate, date_used,
        source). Raises FxRateNotFoundError if none is fresh enough."""
        window_start = date - timedelta(days=self._stale_days)
        window_end = date + timedelta(days=1)

        preferred = self._source_prefs.get(pair, [])
        available = self._store.available_sources(pair)
        ordered = [s for s in preferred if s in available]
        ordered += [s for s in available if s not in ordered]

        for source in ordered:
            df = self._store.read(pair, source, window_start, window_end)
            if df.height == 0:
                continue
            df = df.filter(df["date"] <= date).sort("date", descending=True)
            if df.height == 0:
                continue
            rate = float(df["rate"][0])
            date_used = df["date"][0]
            return rate, date_used, source

        raise FxRateNotFoundError(
            f"no {pair} rate within {self._stale_days} days of "
            f"{date.date().isoformat()} (sources tried: {ordered or 'none'})"
        )
