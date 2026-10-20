"""
test_asset.py

Verifies the Asset value object validates its fields, hashes on asset_id
only, and remains immutable.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import pytest

from locus.domain import Asset, MarketType
from locus.domain.types import AssetId


def _btc() -> Asset:
    return Asset(
        asset_id=AssetId("BTC"),
        symbol="BTCUSDT",
        market_type=MarketType.CRYPTO,
        venue="binance",
        quote_currency="USDT",
    )


def test_asset_constructs() -> None:
    asset = _btc()
    assert asset.asset_id == "BTC"
    assert asset.market_type == MarketType.CRYPTO


def test_asset_is_hashable_by_asset_id() -> None:
    a1 = _btc()
    a2 = Asset(
        asset_id=AssetId("BTC"),
        symbol="XBTUSD",
        market_type=MarketType.CRYPTO,
        venue="kraken",
        quote_currency="USD",
    )
    assert a1 == a2
    assert hash(a1) == hash(a2)
    assert {a1: 1}[a2] == 1


def test_asset_rejects_empty_asset_id() -> None:
    with pytest.raises(ValueError):
        Asset(
            asset_id=AssetId(""),
            symbol="X",
            market_type=MarketType.CRYPTO,
            venue="v",
            quote_currency="USD",
        )


def test_asset_rejects_empty_quote_currency() -> None:
    with pytest.raises(ValueError):
        Asset(
            asset_id=AssetId("BTC"),
            symbol="BTCUSDT",
            market_type=MarketType.CRYPTO,
            venue="binance",
            quote_currency="",
        )


def test_asset_is_immutable() -> None:
    asset = _btc()
    with pytest.raises((AttributeError, Exception)):
        asset.symbol = "ETHUSDT"  # type: ignore[misc]
