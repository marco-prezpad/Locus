"""
perp_types.py

Point-in-time perpetual futures universe: listing records and snapshots.

Author: Marco Pérez Padilla
Date:   23-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class PerpListing:
    """Record of when a perpetual was tradable on a specific venue.

    `delist_date is None` means still live.
    """

    asset_id: str
    venue: str
    list_date: datetime
    delist_date: datetime | None = None

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id must be non-empty")
        if not self.venue:
            raise ValueError("venue must be non-empty")
        if self.list_date.tzinfo is None:
            raise ValueError("list_date must be timezone-aware")
        if self.delist_date is not None:
            if self.delist_date.tzinfo is None:
                raise ValueError("delist_date must be timezone-aware")
            if self.delist_date <= self.list_date:
                raise ValueError("delist_date must be after list_date")


@dataclass(frozen=True, slots=True)
class PerpUniverseSnapshot:
    """Active perpetuals at a specific timestamp."""

    timestamp: datetime
    assets: list[str]
    venue: str
