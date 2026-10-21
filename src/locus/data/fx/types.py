"""
types.py

FX data-layer types. A pair is identified by concatenating base and quote
in ISO style ("EURUSD" = 1 EUR = X USD, "USDTUSD" = 1 USDT = X USD). The
rate semantics are always "1 unit of base = rate units of quote".

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


class FxError(Exception):
    """Base FX error."""


class FxProviderError(FxError):
    """Provider failed to return rates."""


class FxRateNotFoundError(FxError):
    """No rate available for the requested pair/date."""


@dataclass(frozen=True, slots=True)
class FxPair:
    """A currency pair, expressed as base/quote."""

    base: str
    quote: str

    def __post_init__(self) -> None:
        if not self.base or not self.quote:
            raise ValueError("base and quote must be non-empty")
        if not self.base.isupper() or not self.quote.isupper():
            raise ValueError("base and quote must be uppercase")
        if self.base == self.quote:
            raise ValueError("base and quote must differ")

    @property
    def name(self) -> str:
        return f"{self.base}{self.quote}"

    @classmethod
    def parse(cls, name: str) -> FxPair:
        """Parse 'EURUSD' into FxPair('EUR', 'USD').

        Convention: base is the first 3 characters, quote is the rest.
        Sufficient for the MVP pairs (EURUSD, USDTUSD); a general
        currency registry would be needed for arbitrary pairs.
        """
        if len(name) < 4:
            raise ValueError(f"invalid pair name: {name!r}")
        return cls(base=name[:3], quote=name[3:])


@dataclass(frozen=True, slots=True)
class FxRate:
    """A daily FX rate: 1 unit of pair.base = `rate` units of pair.quote."""

    pair: str
    date: datetime
    rate: float

    def __post_init__(self) -> None:
        if not self.pair:
            raise ValueError("pair must be non-empty")
        if self.date.tzinfo is None:
            raise ValueError("date must be timezone-aware")
        if self.rate <= 0:
            raise ValueError("rate must be strictly positive")
