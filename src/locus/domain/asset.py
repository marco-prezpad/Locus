"""
asset.py

The Asset value object: identity, market type, and the minimum metadata
required downstream. Vendor-specific metadata stays in the data layer.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass, field

from locus.domain.enums import MarketType
from locus.domain.types import AssetId


@dataclass(frozen=True, slots=True)
class Asset:
    """A tradable instrument on a specific venue.

    Equality and hashing are based on `asset_id` only. Two assets with the
    same identifier are considered the same instrument regardless of how a
    particular vendor spells the symbol or which quote currency is used.
    """

    asset_id: AssetId
    symbol: str = field(compare=False)
    market_type: MarketType = field(compare=False)
    venue: str = field(compare=False)
    quote_currency: str = field(compare=False)
    metadata: dict[str, str] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id must be non-empty")
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if not self.venue:
            raise ValueError("venue must be non-empty")
        if not self.quote_currency:
            raise ValueError("quote_currency must be non-empty")
