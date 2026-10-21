"""
types.py

Data-layer primitives shared across providers, validation, and storage.
These types are specific to the data layer and must not leak into the
domain (the domain only knows about OHLCVBar, not about HTTP requests
or provider-specific errors).

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Timeframe(StrEnum):
    """Supported bar timeframes."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class ProviderError(Exception):
    """Base error for market data providers."""


class ProviderRateLimitError(ProviderError):
    """Rate limit hit and retries exhausted."""


class ProviderResponseError(ProviderError):
    """Unexpected response shape or values."""


class ProviderNoDataError(ProviderError):
    """Requested range has no data."""


@dataclass(frozen=True, slots=True)
class BarRequest:
    """A request for OHLCV bars to a specific provider.

    `asset_id` is the canonical identifier (e.g. "BTC"). `symbol` is the
    provider-specific ticker (e.g. "BTCUSDT" for Binance). The caller is
    responsible for supplying both; a SymbolMapper will be added in a
    later sub-phase to translate between them automatically.
    """

    asset_id: str
    symbol: str
    timeframe: Timeframe
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id must be non-empty")
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware")
        if self.end <= self.start:
            raise ValueError("end must be strictly greater than start")
