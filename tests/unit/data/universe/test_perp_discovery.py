"""
test_perp_discovery.py

PerpUniverseDiscovery filters and onboardDate parsing.
"""

from __future__ import annotations

import httpx
from pytest_httpx import HTTPXMock

from locus.data.universe.perp_discovery import PerpUniverseDiscovery
from locus.infrastructure.config import FundingConfig


def _discovery() -> PerpUniverseDiscovery:
    return PerpUniverseDiscovery(
        config=FundingConfig(),
        client=httpx.Client(base_url="https://fapi.binance.com"),
    )


def _entry(
    symbol: str,
    contract_type: str = "PERPETUAL",
    underlying_type: str = "COIN",
    status: str = "TRADING",
    onboard_ms: int | None = 1567965300000,
) -> dict:
    e: dict = {
        "symbol": symbol,
        "contractType": contract_type,
        "underlyingType": underlying_type,
        "status": status,
    }
    if onboard_ms is not None:
        e["onboardDate"] = onboard_ms
    return e


def test_discovers_canonical_perp(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        json={"symbols": [_entry("BTCUSDT")]},
    )
    listings = _discovery().discover(["BTC"])
    assert len(listings) == 1
    assert listings[0].asset_id == "BTC"
    assert listings[0].venue == "binance_futures"
    assert listings[0].delist_date is None


def test_filters_tradifi_perp(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        json={"symbols": [_entry("BTCUSDT", contract_type="TRADIFI_PERPETUAL")]},
    )
    assert _discovery().discover(["BTC"]) == []


def test_filters_next_quarter(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        json={"symbols": [_entry("BTCUSDT_270326", contract_type="NEXT_QUARTER")]},
    )
    assert _discovery().discover(["BTC"]) == []


def test_filters_non_coin_underlying(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        json={"symbols": [_entry("BTCUSDT", underlying_type="EQUITY")]},
    )
    assert _discovery().discover(["BTC"]) == []


def test_filters_non_trading(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        json={"symbols": [_entry("BTCUSDT", status="BREAK")]},
    )
    assert _discovery().discover(["BTC"]) == []


def test_skips_missing_onboard_date(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        json={"symbols": [_entry("BTCUSDT", onboard_ms=None)]},
    )
    assert _discovery().discover(["BTC"]) == []


def test_unknown_symbol_skipped(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        json={"symbols": [_entry("FOOUSDT")]},
    )
    assert _discovery().discover(["BTC"]) == []
