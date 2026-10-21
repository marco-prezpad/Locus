"""
types.py

Data-layer types for universe reconstruction: listing records, filter
constraints, and snapshots. These types stay in the data layer; the
builder converts them to domain `Asset` objects at the boundary.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from locus.domain import Asset


@dataclass(frozen=True, slots=True)
class AssetListing:
    """Record of when an asset was tradable on a specific venue.

    `last_tradable_ts is None` means the asset is still trading. In the
    MVP all registry entries are live assets (no delisted symbols are
    discoverable through public APIs); the field exists so the contract
    does not need to change when delisted assets are added.
    """

    asset_id: str
    venue: str
    first_tradable_ts: datetime
    last_tradable_ts: datetime | None = None

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id must be non-empty")
        if not self.venue:
            raise ValueError("venue must be non-empty")
        if self.first_tradable_ts.tzinfo is None:
            raise ValueError("first_tradable_ts must be timezone-aware")
        if self.last_tradable_ts is not None:
            if self.last_tradable_ts.tzinfo is None:
                raise ValueError("last_tradable_ts must be timezone-aware")
            if self.last_tradable_ts <= self.first_tradable_ts:
                raise ValueError("last_tradable_ts must be strictly greater than first_tradable_ts")


@dataclass(frozen=True, slots=True)
class UniverseConstraint:
    """Filters applied when building a universe snapshot.

    `min_history_days` excludes assets that have been listed for less
    than N days at the snapshot timestamp (avoids short-history noise).
    `venues` restricts the snapshot to assets tradable on at least one
    of the given venues.
    """

    min_history_days: int = 30
    venues: list[str] = field(default_factory=lambda: ["binance"])

    def __post_init__(self) -> None:
        if self.min_history_days < 0:
            raise ValueError("min_history_days must be non-negative")
        if not self.venues:
            raise ValueError("venues must be non-empty")


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    """The set of assets tradable at a specific timestamp, given the
    constraints applied at build time."""

    timestamp: datetime
    assets: list[Asset]
    constraints: UniverseConstraint
