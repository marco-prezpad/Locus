"""
market_data.py

Market data primitives: OHLCV bars, trades, and order-book snapshots.
Every instance validates its own invariants at construction time so that
invalid data cannot enter the domain and propagate.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass

from locus.domain.enums import OrderSide
from locus.domain.types import AssetId, Timestamp


@dataclass(frozen=True, slots=True)
class OHLCVBar:
    """A single OHLCV bar for one asset over one interval.

    Invariants enforced at construction:
      - `end` strictly greater than `start`
      - all prices strictly positive
      - `high` at least as large as `open`, `close`, and `low`
      - `low` at most as small as `open`, `close`, and `high`
      - `volume` non-negative
    """

    asset_id: AssetId
    start: Timestamp
    end: Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("bar end must be strictly greater than bar start")
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("all OHLC prices must be strictly positive")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be >= open, close and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be <= open, close and high")
        if self.volume < 0:
            raise ValueError("volume must be non-negative")


@dataclass(frozen=True, slots=True)
class Trade:
    """A single trade print: price, quantity, side, and timestamp."""

    asset_id: AssetId
    timestamp: Timestamp
    price: float
    quantity: float
    side: OrderSide

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError("trade price must be strictly positive")
        if self.quantity <= 0:
            raise ValueError("trade quantity must be strictly positive")


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """Top-of-book snapshot: best bid and best ask with their sizes."""

    asset_id: AssetId
    timestamp: Timestamp
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float

    def __post_init__(self) -> None:
        if self.bid_price <= 0 or self.ask_price <= 0:
            raise ValueError("prices must be strictly positive")
        if self.bid_price >= self.ask_price:
            raise ValueError("bid price must be strictly less than ask price")
        if self.bid_size < 0 or self.ask_size < 0:
            raise ValueError("sizes must be non-negative")

    @property
    def spread(self) -> float:
        """Absolute spread between best ask and best bid."""
        return self.ask_price - self.bid_price
