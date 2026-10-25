"""
funding_base.py

Canonical schema and Protocol for funding providers. Every funding
provider returns a polars DataFrame with exactly FUNDING_SCHEMA columns.

Author: Marco Pérez Padilla
Date:   23-10-2026
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

import polars as pl

FUNDING_SCHEMA: dict[str, Any] = {
    "asset_id": pl.Utf8,
    "venue": pl.Utf8,
    "symbol": pl.Utf8,
    "timestamp": pl.Datetime("us", "UTC"),
    "funding_rate": pl.Float64,
}


class FundingProvider(Protocol):
    """Interface every funding provider must implement."""

    @property
    def venue(self) -> str: ...

    def fetch_funding(self, asset_id: str, start: datetime, end: datetime) -> pl.DataFrame: ...
