"""
test_discovery.py

BinanceUniverseDiscovery tests with mocked HTTP. Covers both listing-date
sources (onboardDate in exchangeInfo, fallback to first 1d bar) and the
error/skip paths.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from pytest_httpx import HTTPXMock

from locus.data.types import ProviderResponseError
from locus.data.universe.discovery import BinanceUniverseDiscovery
from locus.infrastructure.config import BinanceProviderConfig


def _provider() -> BinanceUniverseDiscovery:
    return BinanceUniverseDiscovery(
        config=BinanceProviderConfig(),
        client=httpx.Client(base_url="https://api.binance.com"),
    )


def _exchange_info(symbols: list[dict]) -> dict:
    return {"symbols": symbols}


def _symbol_entry(symbol: str, status: str = "TRADING", onboard_date: int | None = None) -> dict:
    entry: dict = {"symbol": symbol, "status": status}
    if onboard_date is not None:
        entry["onboardDate"] = onboard_date
    return entry


def _kline_response(open_ms: int) -> list[list]:
    return [
        [
            open_ms,
            "100.0",
            "102.0",
            "99.0",
            "101.0",
            "10.5",
            open_ms + 86_399_999,
            "1050.0",
            42,
            "5.0",
            "500.0",
            "0",
        ]
    ]


def _klines_url(symbol: str) -> str:
    return f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&startTime=0&limit=1"


def test_discover_uses_onboard_date_when_present(httpx_mock: HTTPXMock) -> None:
    onboard_ms = int(datetime(2019, 9, 10, tzinfo=UTC).timestamp() * 1000)
    httpx_mock.add_response(
        url="https://api.binance.com/api/v3/exchangeInfo",
        json=_exchange_info([_symbol_entry("BTCUSDT", "TRADING", onboard_ms)]),
    )
    # No klines mock on purpose: if the fallback fires, pytest-httpx
    # will fail the test with "no response found". That proves the
    # onboardDate path was used without hitting klines.
    provider = _provider()
    listings = provider.discover(["BTC"])

    assert len(listings) == 1
    assert listings[0].asset_id == "BTC"
    assert listings[0].venue == "binance"
    assert listings[0].first_tradable_ts == datetime(2019, 9, 10, tzinfo=UTC)


def test_discover_falls_back_to_first_bar(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.binance.com/api/v3/exchangeInfo",
        json=_exchange_info([_symbol_entry("BTCUSDT", "TRADING")]),
    )
    onboard_ms = int(datetime(2019, 9, 10, tzinfo=UTC).timestamp() * 1000)
    httpx_mock.add_response(
        url=_klines_url("BTCUSDT"),
        json=_kline_response(onboard_ms),
    )

    provider = _provider()
    listings = provider.discover(["BTC"])

    assert len(listings) == 1
    assert listings[0].first_tradable_ts == datetime(2019, 9, 10, tzinfo=UTC)


def test_discover_falls_back_when_symbol_not_in_exchange_info(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="https://api.binance.com/api/v3/exchangeInfo",
        json=_exchange_info([]),
    )
    onboard_ms = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp() * 1000)
    httpx_mock.add_response(
        url=_klines_url("BTCUSDT"),
        json=_kline_response(onboard_ms),
    )

    provider = _provider()
    listings = provider.discover(["BTC"])

    assert len(listings) == 1
    assert listings[0].first_tradable_ts == datetime(2020, 1, 1, tzinfo=UTC)


def test_discover_skips_asset_with_no_data(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.binance.com/api/v3/exchangeInfo",
        json=_exchange_info([]),
    )
    httpx_mock.add_response(url=_klines_url("BTCUSDT"), json=[])

    provider = _provider()
    listings = provider.discover(["BTC"])

    assert listings == []


def test_discover_non_trading_status_falls_back(httpx_mock: HTTPXMock) -> None:
    """A symbol whose status is not TRADING is not trusted from
    exchangeInfo. We fall back to the first bar to get a usable date."""
    httpx_mock.add_response(
        url="https://api.binance.com/api/v3/exchangeInfo",
        json=_exchange_info([_symbol_entry("BTCUSDT", "BREAK", onboard_date=0)]),
    )
    onboard_ms = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp() * 1000)
    httpx_mock.add_response(
        url=_klines_url("BTCUSDT"),
        json=_kline_response(onboard_ms),
    )

    provider = _provider()
    listings = provider.discover(["BTC"])

    assert len(listings) == 1
    assert listings[0].first_tradable_ts == datetime(2020, 1, 1, tzinfo=UTC)


def test_discover_malformed_exchange_info_raises(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.binance.com/api/v3/exchangeInfo",
        json={"foo": "bar"},
    )

    provider = _provider()
    with pytest.raises(ProviderResponseError):
        provider.discover(["BTC"])


def test_discover_unknown_asset_raises_before_http(httpx_mock: HTTPXMock) -> None:
    """If the SymbolMapper does not know the asset, discover() fails fast
    before any HTTP call. No mock is needed; the fixture just ensures no
    accidental network access."""
    provider = _provider()
    with pytest.raises(ProviderResponseError):
        provider.discover(["DOES_NOT_EXIST"])


def test_discover_multiple_assets_mixed(httpx_mock: HTTPXMock) -> None:
    """Mixed scenario: BTC via onboardDate, ETH via fallback, SOL via
    fallback because it's not in exchangeInfo."""
    onboard_btc_ms = int(datetime(2019, 9, 10, tzinfo=UTC).timestamp() * 1000)
    httpx_mock.add_response(
        url="https://api.binance.com/api/v3/exchangeInfo",
        json=_exchange_info(
            [
                _symbol_entry("BTCUSDT", "TRADING", onboard_btc_ms),
                _symbol_entry("ETHUSDT", "TRADING"),
            ]
        ),
    )
    onboard_eth_ms = int(datetime(2020, 1, 15, tzinfo=UTC).timestamp() * 1000)
    httpx_mock.add_response(
        url=_klines_url("ETHUSDT"),
        json=_kline_response(onboard_eth_ms),
    )
    onboard_sol_ms = int(datetime(2020, 8, 11, tzinfo=UTC).timestamp() * 1000)
    httpx_mock.add_response(
        url=_klines_url("SOLUSDT"),
        json=_kline_response(onboard_sol_ms),
    )

    provider = _provider()
    listings = provider.discover(["BTC", "ETH", "SOL"])

    assert len(listings) == 3
    by_asset = {list.asset_id: list.first_tradable_ts for list in listings}
    assert by_asset["BTC"] == datetime(2019, 9, 10, tzinfo=UTC)
    assert by_asset["ETH"] == datetime(2020, 1, 15, tzinfo=UTC)
    assert by_asset["SOL"] == datetime(2020, 8, 11, tzinfo=UTC)
