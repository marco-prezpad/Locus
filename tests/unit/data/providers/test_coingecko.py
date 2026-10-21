"""
test_coingecko.py

Unit tests for CoinGeckoMarketDataProvider.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from pytest_httpx import HTTPXMock

from locus.data.providers.base import BAR_SCHEMA
from locus.data.providers.coingecko import CoinGeckoMarketDataProvider
from locus.data.types import BarRequest, ProviderResponseError, Timeframe
from locus.infrastructure.config import CoinGeckoProviderConfig


def _provider() -> CoinGeckoMarketDataProvider:
    return CoinGeckoMarketDataProvider(
        config=CoinGeckoProviderConfig(),
        client=httpx.Client(base_url="https://api.coingecko.com/api/v3"),
    )


def _request_daily() -> BarRequest:
    return BarRequest(
        asset_id="BTC",
        timeframe=Timeframe.D1,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 4, tzinfo=UTC),
    )


def test_daily_bars_returned(httpx_mock: HTTPXMock) -> None:
    base_ms = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000)
    httpx_mock.add_response(
        json=[
            [base_ms, 100.0, 110.0, 95.0, 105.0],
            [base_ms + 86_400_000, 105.0, 115.0, 100.0, 110.0],
            [base_ms + 2 * 86_400_000, 110.0, 120.0, 108.0, 118.0],
        ]
    )
    provider = _provider()
    df = provider.fetch_bars(_request_daily())
    assert df.schema == BAR_SCHEMA
    assert df.height == 3
    assert df["volume"].to_list() == [0.0, 0.0, 0.0]


def test_intraday_timeframe_raises() -> None:
    provider = _provider()
    bad = BarRequest(
        asset_id="BTC",
        timeframe=Timeframe.H1,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
    )
    with pytest.raises(ProviderResponseError):
        provider.fetch_bars(bad)


def test_venue_is_coingecko() -> None:
    assert _provider().venue == "coingecko"
