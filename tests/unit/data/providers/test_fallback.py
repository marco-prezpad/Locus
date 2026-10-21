"""
test_fallback.py

FallbackMarketDataProvider tries providers in order; verifies success
cases, error propagation, and empty-data behavior.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.data.providers.base import BAR_SCHEMA
from locus.data.providers.fallback import FallbackMarketDataProvider
from locus.data.types import BarRequest, ProviderError, Timeframe


def _df(n: int = 1) -> pl.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        {
            "asset_id": "",
            "venue": "fake",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "timestamp": base + timedelta(hours=i),
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 10.0,
        }
        for i in range(n)
    ]
    return pl.DataFrame(rows, schema=BAR_SCHEMA)


class _SuccessProvider:
    venue = "success"

    def fetch_bars(self, request: BarRequest) -> pl.DataFrame:
        return _df(2)


class _FailingProvider:
    venue = "failing"

    def fetch_bars(self, request: BarRequest) -> pl.DataFrame:
        raise ProviderError("boom")


class _EmptyProvider:
    venue = "empty"

    def fetch_bars(self, request: BarRequest) -> pl.DataFrame:
        return pl.DataFrame(schema=BAR_SCHEMA)


def _request() -> BarRequest:
    return BarRequest(
        asset_id="BTC",
        timeframe=Timeframe.H1,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
    )


def test_first_provider_wins() -> None:
    chain = FallbackMarketDataProvider([_SuccessProvider(), _FailingProvider()])
    df = chain.fetch_bars(_request())
    assert df.height == 2


def test_second_provider_used_on_failure() -> None:
    chain = FallbackMarketDataProvider([_FailingProvider(), _SuccessProvider()])
    df = chain.fetch_bars(_request())
    assert df.height == 2


def test_second_provider_used_on_empty() -> None:
    chain = FallbackMarketDataProvider([_EmptyProvider(), _SuccessProvider()])
    df = chain.fetch_bars(_request())
    assert df.height == 2


def test_all_fail_raises() -> None:
    chain = FallbackMarketDataProvider([_FailingProvider(), _FailingProvider()])
    with pytest.raises(ProviderError):
        chain.fetch_bars(_request())


def test_all_empty_returns_empty() -> None:
    chain = FallbackMarketDataProvider([_EmptyProvider(), _EmptyProvider()])
    df = chain.fetch_bars(_request())
    assert df.height == 0


def test_empty_provider_list_raises() -> None:
    with pytest.raises(ValueError):
        FallbackMarketDataProvider([])
