"""
test_perp_registry.py

PerpUniverseRegistry persistence and point-in-time lookup.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from locus.data.universe.perp_registry import PerpUniverseRegistry
from locus.data.universe.perp_types import PerpListing


def _listing(
    asset: str = "BTC",
    list_ts: datetime | None = None,
    delist_ts: datetime | None = None,
) -> PerpListing:
    return PerpListing(
        asset_id=asset,
        venue="binance_futures",
        list_date=list_ts or datetime(2020, 1, 1, tzinfo=UTC),
        delist_date=delist_ts,
    )


def test_is_listed_before_list_date() -> None:
    reg = PerpUniverseRegistry([_listing()])
    assert not reg.is_listed("BTC", "binance_futures", datetime(2019, 12, 31, tzinfo=UTC))


def test_is_listed_within_window() -> None:
    reg = PerpUniverseRegistry([_listing()])
    assert reg.is_listed("BTC", "binance_futures", datetime(2023, 6, 1, tzinfo=UTC))


def test_is_listed_after_delist() -> None:
    reg = PerpUniverseRegistry([_listing(delist_ts=datetime(2024, 6, 1, tzinfo=UTC))])
    assert not reg.is_listed("BTC", "binance_futures", datetime(2024, 7, 1, tzinfo=UTC))


def test_unknown_asset_not_listed() -> None:
    reg = PerpUniverseRegistry([_listing()])
    assert not reg.is_listed("FOO", "binance_futures", datetime(2023, 1, 1, tzinfo=UTC))


def test_parquet_roundtrip(tmp_path: Path) -> None:
    reg = PerpUniverseRegistry([_listing("BTC"), _listing("ETH")])
    path = tmp_path / "registry.parquet"
    reg.to_parquet(path)
    loaded = PerpUniverseRegistry.from_parquet(path)
    assert len(loaded) == 2
    assert set(loaded.asset_ids()) == {"BTC", "ETH"}


def test_from_parquet_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PerpUniverseRegistry.from_parquet(tmp_path / "missing.parquet")


def test_universe_point_in_time() -> None:
    """Pre-registration §3 required test."""
    listing = PerpListing(
        asset_id="TEST",
        venue="binance_futures",
        list_date=datetime(2023, 1, 1, tzinfo=UTC),
        delist_date=datetime(2024, 6, 1, tzinfo=UTC),
    )
    reg = PerpUniverseRegistry([listing])
    assert reg.is_listed("TEST", "binance_futures", datetime(2023, 6, 1, tzinfo=UTC))
    assert not reg.is_listed("TEST", "binance_futures", datetime(2024, 7, 1, tzinfo=UTC))
    assert not reg.is_listed("TEST", "binance_futures", datetime(2022, 12, 31, tzinfo=UTC))
