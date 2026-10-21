"""
base.py

FX provider interface and canonical output schema. Every FX provider
returns a polars DataFrame with exactly FX_PROVIDER_SCHEMA columns. The
store enriches it with `source` and `fetched_at` on write.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

import polars as pl

FX_PROVIDER_SCHEMA: dict[str, Any] = {
    "pair": pl.Utf8,
    "date": pl.Datetime("us", "UTC"),
    "rate": pl.Float64,
}


class FxRateProvider(Protocol):
    """Interface every FX provider must implement.

    Implementations return a polars DataFrame with FX_PROVIDER_SCHEMA
    columns, sorted by `date`, with no duplicates. An empty DataFrame
    with the right schema is a valid response if the range has no data.
    """

    @property
    def source(self) -> str:
        """Stable identifier of the source (e.g. "frankfurter")."""
        ...

    def fetch_rates(self, pair: str, start: datetime, end: datetime) -> pl.DataFrame:
        """Return daily rates for `pair` in [start, end).

        `start` and `end` must be timezone-aware. Returns a DataFrame
        with FX_PROVIDER_SCHEMA columns.
        """
        ...
