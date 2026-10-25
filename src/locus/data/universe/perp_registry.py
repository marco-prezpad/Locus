"""
perp_registry.py

Parquet-backed registry of perpetual listings.

Layout: data/processed/universe_perps/registry.parquet

Author: Marco Pérez Padilla
Date:   23-10-2026
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from locus.data.universe.perp_types import PerpListing

_PERP_REGISTRY_SCHEMA: dict[str, Any] = {
    "asset_id": pl.Utf8,
    "venue": pl.Utf8,
    "list_date": pl.Datetime("us", "UTC"),
    "delist_date": pl.Datetime("us", "UTC"),
}


class PerpUniverseRegistry:
    """Query and persist perp listings."""

    def __init__(self, listings: list[PerpListing]) -> None:
        self._by_key: dict[tuple[str, str], PerpListing] = {
            (list.asset_id, list.venue): list for list in listings
        }
        self._listings = listings

    def __len__(self) -> int:
        return len(self._listings)

    def all(self) -> list[PerpListing]:
        return list(self._listings)

    def is_listed(self, asset_id: str, venue: str, timestamp: datetime) -> bool:
        listing = self._by_key.get((asset_id, venue))
        if listing is None:
            return False
        if timestamp < listing.list_date:
            return False
        return not (listing.delist_date is not None and timestamp >= listing.delist_date)

    def asset_ids(self) -> list[str]:
        return sorted({list.asset_id for list in self._listings})

    def to_parquet(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "asset_id": list.asset_id,
                "venue": list.venue,
                "list_date": list.list_date,
                "delist_date": list.delist_date,
            }
            for list in self._listings
        ]
        df = pl.DataFrame(rows, schema=_PERP_REGISTRY_SCHEMA).sort(["asset_id", "venue"])
        df.write_parquet(path, compression="snappy")

    @classmethod
    def from_parquet(cls, path: Path) -> PerpUniverseRegistry:
        if not path.exists():
            raise FileNotFoundError(f"Perp registry not found: {path}")
        df = pl.read_parquet(path)
        listings = [
            PerpListing(
                asset_id=row["asset_id"],
                venue=row["venue"],
                list_date=row["list_date"],
                delist_date=row["delist_date"],
            )
            for row in df.iter_rows(named=True)
        ]
        return cls(listings)
