"""
test_builder.py

UniverseBuilder applies listing windows and constraints correctly.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from locus.data.universe.builder import UniverseBuilder
from locus.data.universe.registry import UniverseRegistry
from locus.data.universe.types import (
    AssetListing,
    UniverseConstraint,
)


def _ts(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _registry() -> UniverseRegistry:
    return UniverseRegistry(
        [
            AssetListing(
                asset_id="BTC",
                venue="binance",
                first_tradable_ts=_ts(2019, 9, 10),
            ),
            AssetListing(
                asset_id="ETH",
                venue="binance",
                first_tradable_ts=_ts(2020, 1, 15),
            ),
            AssetListing(
                asset_id="NEW",
                venue="binance",
                first_tradable_ts=_ts(2025, 1, 1),
            ),
        ]
    )


def test_universe_at_timestamp_includes_only_listed() -> None:
    b = UniverseBuilder(_registry())
    snap = b.build(_ts(2021, 1, 1))
    asset_ids = sorted(a.asset_id for a in snap.assets)
    assert asset_ids == ["BTC", "ETH"]


def test_universe_before_any_listing_empty() -> None:
    b = UniverseBuilder(_registry())
    snap = b.build(_ts(2019, 1, 1))
    assert snap.assets == []


def test_universe_includes_assets_listed_for_min_history() -> None:
    b = UniverseBuilder(_registry())
    snap = b.build(_ts(2021, 1, 1), UniverseConstraint(min_history_days=300))
    asset_ids = sorted(a.asset_id for a in snap.assets)
    # BTC listed 2019-09-10 (>300d before 2021-01-01)
    # ETH listed 2020-01-15 (352d before 2021-01-01, >300d)
    # NEW listed 2025-01-01 (after 2021-01-01, excluded)
    assert asset_ids == ["BTC", "ETH"]


def test_universe_min_history_excludes_recent() -> None:
    b = UniverseBuilder(_registry())
    snap = b.build(_ts(2025, 2, 1), UniverseConstraint(min_history_days=60))
    asset_ids = sorted(a.asset_id for a in snap.assets)
    # NEW listed 2025-01-01, only 31 days before 2025-02-01
    assert asset_ids == ["BTC", "ETH"]


def test_universe_rejects_naive_timestamp() -> None:
    b = UniverseBuilder(_registry())
    with pytest.raises(ValueError):
        b.build(datetime(2021, 1, 1))


def test_universe_assets_are_domain_objects() -> None:
    from locus.domain import Asset, MarketType

    b = UniverseBuilder(_registry())
    snap = b.build(_ts(2021, 1, 1))
    for asset in snap.assets:
        assert isinstance(asset, Asset)
        assert asset.market_type == MarketType.CRYPTO
        assert asset.venue == "binance"
        assert asset.quote_currency == "USDT"
        assert asset.symbol.endswith("USDT")
