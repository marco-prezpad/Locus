"""
base.py

Provider interface and canonical OHLCV schema. Every market data
provider must conform to this Protocol and return a polars DataFrame
with exactly the BAR_SCHEMA columns.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from typing import Any, Protocol

import polars as pl

from locus.data.types import BarRequest

BAR_SCHEMA: dict[str, Any] = {
    "asset_id": pl.Utf8,
    "venue": pl.Utf8,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "timestamp": pl.Datetime("us", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


class MarketDataProvider(Protocol):
    """Interface every market data provider must implement.

    Implementations return a polars DataFrame with the canonical
    BAR_SCHEMA columns. All timestamps must be timezone-aware UTC and
    represent the bar open time.
    """

    @property
    def venue(self) -> str:
        """Stable identifier of the venue (e.g. "binance")."""
        ...

    def fetch_bars(self, request: BarRequest) -> pl.DataFrame:
        """Fetch bars for the given request.

        Implementations must paginate internally if needed and return all
        bars in [request.start, request.end), sorted by timestamp, with no
        duplicates. Returns an empty DataFrame (with the right schema) if
        the range has no data.
        """
        ...
