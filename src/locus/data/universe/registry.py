"""
registry.py

In-memory registry of asset listings with Parquet persistence. The
registry is the source of truth for UniverseBuilder: given a timestamp
and constraints, the builder asks the registry which assets were listed.

Parquet layout: a single file at `path` with one row per (asset, venue).
Writing overwrites the file (registry is small, ~150 rows for the MVP).

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from locus.data.universe.types import AssetListing

_LISTING_SCHEMA: dict[str, Any] = {
    "asset_id": pl.Utf8,
    "venue": pl.Utf8,
    "first_tradable_ts": pl.Datetime("us", "UTC"),
    "last_tradable_ts": pl.Datetime("us", "UTC"),
}


class UniverseRegistry:
    """Query and persist asset listings."""

    def __init__(self, listings: list[AssetListing]) -> None:
        self._by_key: dict[tuple[str, str], AssetListing] = {
            (list.asset_id, list.venue): list for list in listings
        }
        self._listings = listings

    def __len__(self) -> int:
        return len(self._listings)

    def all(self) -> list[AssetListing]:
        """Return every listing in the registry, unsorted."""
        return list(self._listings)

    def is_listed(self, asset_id: str, venue: str, timestamp: datetime) -> bool:
        """True if `asset_id` was tradable on `venue` at `timestamp`."""
        listing = self._by_key.get((asset_id, venue))
        if listing is None:
            return False
        if timestamp < listing.first_tradable_ts:
            return False
        return not (listing.last_tradable_ts is not None and timestamp >= listing.last_tradable_ts)

    def asset_ids(self) -> list[str]:
        """Unique asset ids present in the registry."""
        return sorted({list.asset_id for list in self._listings})

    def venues(self) -> list[str]:
        """Unique venues present in the registry."""
        return sorted({list.venue for list in self._listings})

    def to_parquet(self, path: Path) -> None:
        """Persist the registry as a single Parquet file (overwrites)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "asset_id": list.asset_id,
                "venue": list.venue,
                "first_tradable_ts": list.first_tradable_ts,
                "last_tradable_ts": list.last_tradable_ts,
            }
            for list in self._listings
        ]
        df = pl.DataFrame(rows, schema=_LISTING_SCHEMA).sort(["asset_id", "venue"])
        df.write_parquet(path, compression="snappy")

    @classmethod
    def from_parquet(cls, path: Path) -> UniverseRegistry:
        """Load a registry from Parquet. Raises FileNotFoundError if the
        file does not exist."""
        if not path.exists():
            raise FileNotFoundError(f"Universe registry not found: {path}")
        df = pl.read_parquet(path)
        listings = [
            AssetListing(
                asset_id=row["asset_id"],
                venue=row["venue"],
                first_tradable_ts=row["first_tradable_ts"],
                last_tradable_ts=row["last_tradable_ts"],
            )
            for row in df.iter_rows(named=True)
        ]
        return cls(listings)
