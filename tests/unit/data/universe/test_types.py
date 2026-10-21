"""
test_types.py

Invariants for AssetListing, UniverseConstraint, UniverseSnapshot.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from locus.data.universe.types import (
    AssetListing,
    UniverseConstraint,
)


def _ts(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def test_asset_listing_valid() -> None:
    list = AssetListing(asset_id="BTC", venue="binance", first_tradable_ts=_ts(2019, 9, 10))
    assert list.last_tradable_ts is None


def test_asset_listing_rejects_naive_ts() -> None:
    with pytest.raises(ValueError):
        AssetListing(
            asset_id="BTC",
            venue="binance",
            first_tradable_ts=datetime(2019, 9, 10),
        )


def test_asset_listing_rejects_last_before_first() -> None:
    with pytest.raises(ValueError):
        AssetListing(
            asset_id="BTC",
            venue="binance",
            first_tradable_ts=_ts(2020, 1, 1),
            last_tradable_ts=_ts(2019, 1, 1),
        )


def test_universe_constraint_valid() -> None:
    c = UniverseConstraint()
    assert c.min_history_days == 30
    assert c.venues == ["binance"]


def test_universe_constraint_rejects_empty_venues() -> None:
    with pytest.raises(ValueError):
        UniverseConstraint(venues=[])


def test_universe_constraint_rejects_negative_history() -> None:
    with pytest.raises(ValueError):
        UniverseConstraint(min_history_days=-1)
