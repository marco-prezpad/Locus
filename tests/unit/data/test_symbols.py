"""
test_symbols.py

Verifies symbol translation between canonical asset ids and per-venue
symbols.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import pytest

from locus.data.symbols import SymbolMapper
from locus.data.types import ProviderResponseError


def test_btc_mapping() -> None:
    assert SymbolMapper.to_symbol("BTC", "binance") == "BTCUSDT"
    assert SymbolMapper.to_symbol("BTC", "kraken") == "XBTUSDT"
    assert SymbolMapper.to_symbol("BTC", "coingecko") == "bitcoin"


def test_eth_mapping() -> None:
    assert SymbolMapper.to_symbol("ETH", "binance") == "ETHUSDT"
    assert SymbolMapper.to_symbol("ETH", "kraken") == "ETHUSDT"
    assert SymbolMapper.to_symbol("ETH", "coingecko") == "ethereum"


def test_reverse_mapping() -> None:
    assert SymbolMapper.to_asset_id("BTCUSDT", "binance") == "BTC"
    assert SymbolMapper.to_asset_id("XBTUSDT", "kraken") == "BTC"
    assert SymbolMapper.to_asset_id("bitcoin", "coingecko") == "BTC"


def test_unknown_asset_raises() -> None:
    with pytest.raises(ProviderResponseError):
        SymbolMapper.to_symbol("FOO", "binance")


def test_unknown_venue_raises() -> None:
    with pytest.raises(ProviderResponseError):
        SymbolMapper.to_symbol("BTC", "unknown_venue")


def test_unknown_symbol_raises() -> None:
    with pytest.raises(ProviderResponseError):
        SymbolMapper.to_asset_id("FOOUSDT", "binance")


def test_supported_assets_nonempty() -> None:
    assets = SymbolMapper.supported_assets()
    assert "BTC" in assets
    assert len(assets) >= 20


def test_supported_venues() -> None:
    venues = SymbolMapper.supported_venues()
    assert set(venues) >= {"binance", "kraken", "coingecko"}


def test_all_assets_mapped_for_all_venues() -> None:
    """Every canonical asset must have a symbol for every supported venue.

    This is the guardrail that catches "I added FOO but forgot Kraken".
    Fails loudly the moment a new asset is half-mapped.
    """
    venues = SymbolMapper.supported_venues()
    assets = SymbolMapper.supported_assets()
    assert venues, "no venues registered"
    assert assets, "no assets registered"

    missing: list[tuple[str, str]] = []
    for asset in assets:
        for venue in venues:
            try:
                SymbolMapper.to_symbol(asset, venue)
            except ProviderResponseError:
                missing.append((asset, venue))

    assert not missing, "assets missing venue mappings: " + ", ".join(
        f"{a}->{v}" for a, v in missing
    )


def test_no_duplicate_symbols_within_a_venue() -> None:
    """Two different assets must not map to the same symbol in the same
    venue. A duplicate is a silent misrouting bug: the system would buy
    the wrong asset.
    """
    venues = SymbolMapper.supported_venues()
    assets = SymbolMapper.supported_assets()

    for venue in venues:
        seen: dict[str, str] = {}
        for asset in assets:
            try:
                symbol = SymbolMapper.to_symbol(asset, venue)
            except ProviderResponseError:
                continue
            if symbol in seen:
                raise AssertionError(
                    f"venue {venue!r} maps both {seen[symbol]!r} and {asset!r} to symbol {symbol!r}"
                )
            seen[symbol] = asset


def test_reverse_lookup_roundtrip() -> None:
    """For every (asset, venue), to_asset_id(to_symbol(...)) must
    return the original asset id."""
    venues = SymbolMapper.supported_venues()
    assets = SymbolMapper.supported_assets()

    for venue in venues:
        for asset in assets:
            try:
                symbol = SymbolMapper.to_symbol(asset, venue)
            except ProviderResponseError:
                continue
            recovered = SymbolMapper.to_asset_id(symbol, venue)
            assert recovered == asset, (
                f"roundtrip failed for {asset} @ {venue}: {asset} -> {symbol} -> {recovered}"
            )
