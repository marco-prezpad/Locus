"""
test_binance.py

Unit tests for BinanceMarketDataProvider using mocked HTTP responses.
No network calls; safe to run in CI.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from pytest_httpx import HTTPXMock

from locus.data.providers.base import BAR_SCHEMA
from locus.data.providers.binance import BinanceMarketDataProvider
from locus.data.types import BarRequest, ProviderResponseError, Timeframe
from locus.infrastructure.config import BinanceProviderConfig


def _kline(open_ms: int, price: float = 100.0) -> list:
    return [
        open_ms,
        str(price),
        str(price + 2.0),
        str(price - 1.0),
        str(price + 1.0),
        "10.5",
        open_ms + 3_599_999,
        "1050.0",
        42,
        "5.0",
        "500.0",
        "0",
    ]


def _provider(httpx_mock: HTTPXMock) -> BinanceMarketDataProvider:
    client = httpx.Client(base_url="https://api.binance.com")
    return BinanceMarketDataProvider(
        config=BinanceProviderConfig(),
        client=client,
    )


def _request() -> BarRequest:
    return BarRequest(
        asset_id="BTC",
        timeframe=Timeframe.H1,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 1, 2, tzinfo=UTC),
    )


def test_fetch_bars_returns_canonical_schema(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&startTime=1767225600000&endTime=1767232800000&limit=1000",
        json=[
            _kline(1767225600000, 100.0),
            _kline(1767229200000, 101.0),
        ],
    )
    provider = _provider(httpx_mock)
    df = provider.fetch_bars(_request())
    assert df.schema == BAR_SCHEMA
    assert df.height == 2
    assert df["close"][0] == pytest.approx(101.0)


def test_fetch_bars_empty_response(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&startTime=1767225600000&endTime=1767232800000&limit=1000",
        json=[],
    )
    provider = _provider(httpx_mock)
    df = provider.fetch_bars(_request())
    assert df.height == 0
    assert df.schema == BAR_SCHEMA


def test_fetch_bars_paginates(httpx_mock: HTTPXMock) -> None:
    # First batch fills the request limit; second batch is empty.
    httpx_mock.add_response(
        json=[_kline(1767225600000, 100.0)],
    )
    httpx_mock.add_response(json=[])
    provider = _provider(httpx_mock)
    df = provider.fetch_bars(_request())
    assert df.height == 1


def test_venue_is_binance(httpx_mock: HTTPXMock) -> None:
    provider = _provider(httpx_mock)
    assert provider.venue == "binance"


def test_unsupported_timeframe_raises(httpx_mock: HTTPXMock) -> None:
    provider = _provider(httpx_mock)
    bad = BarRequest(
        asset_id="BTC",
        timeframe="2h",  # type: ignore[arg-type]
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
    )
    with pytest.raises(ProviderResponseError):
        provider.fetch_bars(bad)
