"""
test_registry.py

UniverseRegistry query and Parquet round-trip.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from locus.data.universe.registry import UniverseRegistry
from locus.data.universe.types import AssetListing


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
                asset_id="XYZ",
                venue="binance",
                first_tradable_ts=_ts(2022, 1, 1),
                last_tradable_ts=_ts(2024, 1, 1),
            ),
        ]
    )


def test_is_listed_before_first() -> None:
    r = _registry()
    assert not r.is_listed("BTC", "binance", _ts(2019, 1, 1))


def test_is_listed_after_first() -> None:
    r = _registry()
    assert r.is_listed("BTC", "binance", _ts(2020, 1, 1))


def test_is_listed_after_delisting() -> None:
    r = _registry()
    assert not r.is_listed("XYZ", "binance", _ts(2024, 6, 1))


def test_is_listed_unknown_asset() -> None:
    r = _registry()
    assert not r.is_listed("NOPE", "binance", _ts(2020, 1, 1))


def test_asset_ids_sorted() -> None:
    r = _registry()
    assert r.asset_ids() == ["BTC", "ETH", "XYZ"]


def test_venues() -> None:
    r = _registry()
    assert r.venues() == ["binance"]


def test_parquet_roundtrip(tmp_path: Path) -> None:
    r = _registry()
    path = tmp_path / "registry.parquet"
    r.to_parquet(path)
    r2 = UniverseRegistry.from_parquet(path)
    assert r2.asset_ids() == r.asset_ids()
    assert len(r2) == len(r)


def test_from_parquet_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        UniverseRegistry.from_parquet(tmp_path / "missing.parquet")
