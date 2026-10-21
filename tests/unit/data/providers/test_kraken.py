"""
test_kraken.py

Unit tests for KrakenMarketDataProvider using mocked HTTP responses.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from pytest_httpx import HTTPXMock

from locus.data.providers.base import BAR_SCHEMA
from locus.data.providers.kraken import KrakenMarketDataProvider
from locus.data.types import BarRequest, ProviderNoDataError, ProviderResponseError, Timeframe
from locus.infrastructure.config import KrakenProviderConfig


def _kraken_response(bars: list[list], last: int) -> dict:
    return {"error": [], "result": {"XXBTZUSDT": bars, "last": last}}


def _kraken_bar(unix_ts: int, price: float = 100.0) -> list:
    return [
        unix_ts,
        str(price),
        str(price + 2.0),
        str(price - 1.0),
        str(price + 1.0),
        str(price + 0.5),
        "10.5",
        42,
    ]


def _provider() -> KrakenMarketDataProvider:
    return KrakenMarketDataProvider(
        config=KrakenProviderConfig(),
        client=httpx.Client(base_url="https://api.kraken.com"),
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
        json=_kraken_response(
            [
                _kraken_bar(1767225600, 100.0),
                _kraken_bar(1767229200, 101.0),
            ],
            last=1767229200,
        )
    )
    provider = _provider()
    df = provider.fetch_bars(_request())
    assert df.schema == BAR_SCHEMA
    assert df.height == 2


def test_fetch_bars_empty_response(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(json={"error": [], "result": {"last": 1767225600}})
    provider = _provider()
    df = provider.fetch_bars(_request())
    assert df.height == 0
    assert df.schema == BAR_SCHEMA


def test_kraken_error_raises(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(json={"error": ["EQuery:Unknown asset pair"], "result": {}})
    provider = _provider()
    with pytest.raises(ProviderResponseError):
        provider.fetch_bars(_request())


def test_venue_is_kraken() -> None:
    assert _provider().venue == "kraken"


def test_unsupported_timeframe_raises() -> None:
    provider = _provider()
    bad = BarRequest(
        asset_id="BTC",
        timeframe="2h",  # type: ignore[arg-type]
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
    )
    with pytest.raises(ProviderResponseError):
        provider.fetch_bars(bad)


def test_provider_raises_no_data_when_returned_range_exceeds_request(
    httpx_mock: HTTPXMock,
) -> None:
    """Kraken's public API returns the most recent 720 bars regardless of
    `since`. If the first returned bar is after the requested end, the
    provider must raise ProviderNoDataError (not return an empty frame)."""
    httpx_mock.add_response(
        json=_kraken_response(
            [
                _kraken_bar(1893456000, 100.0),  # 2030-01-01, way after request
            ],
            last=1893456000,
        )
    )
    provider = _provider()
    with pytest.raises(ProviderNoDataError):
        provider.fetch_bars(_request())
