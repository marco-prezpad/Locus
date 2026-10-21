"""
builder.py

UniverseBuilder reconstructs the set of tradable assets at any
historical timestamp. The builder reads from a UniverseRegistry and
applies UniverseConstraint filters. Assets are returned as domain
`Asset` objects, ready for downstream phases.

The builder never accesses the network. Discovery (see
`discovery.py`) populates the registry offline; the builder is a pure
function of (registry, timestamp, constraints).

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import datetime, timedelta

from locus.data.symbols import SymbolMapper
from locus.data.universe.registry import UniverseRegistry
from locus.data.universe.types import (
    UniverseConstraint,
    UniverseSnapshot,
)
from locus.domain import Asset, AssetId, MarketType
from locus.infrastructure.logging import LoggerFactory


class UniverseBuilder:
    """Build historical universe snapshots from a listing registry."""

    def __init__(self, registry: UniverseRegistry) -> None:
        self._registry = registry
        self._log = LoggerFactory.get("data.universe.builder")

    def build(
        self,
        timestamp: datetime,
        constraints: UniverseConstraint | None = None,
    ) -> UniverseSnapshot:
        """Return the universe available at `timestamp` under constraints.

        An asset is included if, for at least one venue in
        `constraints.venues`, the registry reports it as listed at
        `timestamp` and it has been listed for at least
        `constraints.min_history_days`.
        """
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        c = constraints or UniverseConstraint()
        min_listed_since = timestamp - timedelta(days=c.min_history_days)

        assets: list[Asset] = []
        for asset_id in self._registry.asset_ids():
            for venue in c.venues:
                if not self._registry.is_listed(asset_id, venue, timestamp):
                    continue
                if not self._registry.is_listed(asset_id, venue, min_listed_since):
                    continue
                try:
                    symbol = SymbolMapper.to_symbol(asset_id, venue)
                except Exception:
                    continue
                assets.append(
                    Asset(
                        asset_id=AssetId(asset_id),
                        symbol=symbol,
                        market_type=MarketType.CRYPTO,
                        venue=venue,
                        quote_currency="USDT",
                    )
                )
                break

        self._log.info(
            "universe_built",
            extra={
                "assets": len(assets),
                "venues": ",".join(c.venues),
                "timestamp": timestamp.isoformat(),
            },
        )
        return UniverseSnapshot(
            timestamp=timestamp,
            assets=assets,
            constraints=c,
        )
