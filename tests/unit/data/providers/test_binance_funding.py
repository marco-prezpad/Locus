"""
test_binance_funding.py

Unit tests for BinanceFundingProvider using mocked HTTP.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from pytest_httpx import HTTPXMock

from locus.data.providers.binance_funding import BinanceFundingProvider
from locus.data.providers.funding_base import FUNDING_SCHEMA
from locus.data.types import ProviderResponseError
from locus.infrastructure.config import FundingConfig


def _provider() -> BinanceFundingProvider:
    return BinanceFundingProvider(
        config=FundingConfig(),
        client=httpx.Client(base_url="https://fapi.binance.com"),
    )


def _req() -> tuple[datetime, datetime]:
    return (
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
    )


def _event(ms: int, rate: str = "0.00010000") -> dict:
    return {
        "symbol": "BTCUSDT",
        "fundingTime": ms,
        "fundingRate": rate,
        "markPrice": "50000.0",
        "rateType": "Regular",
    }


def test_fetch_funding_returns_canonical_schema(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        json=[_event(1767225600000), _event(1767254400000)],
    )
    start, end = _req()
    df = _provider().fetch_funding("BTC", start, end)
    assert df.schema == FUNDING_SCHEMA
    assert df.height == 2
    assert df["funding_rate"][0] == pytest.approx(0.0001)


def test_fetch_funding_with_raw_returns_batch_records(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(json=[_event(1767225600000)])
    start, end = _req()
    df, raw_records = _provider().fetch_funding_with_raw("BTC", start, end)
    assert df.height == 1
    assert len(raw_records) == 1
    assert raw_records[0]["symbol"] == "BTCUSDT"
    assert len(raw_records[0]["response"]) == 1


def test_fetch_funding_empty_response(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(json=[])
    start, end = _req()
    df = _provider().fetch_funding("BTC", start, end)
    assert df.height == 0
    assert df.schema == FUNDING_SCHEMA


def test_fetch_funding_rejects_non_list_response(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(json={"error": "rate limit"})
    start, end = _req()
    with pytest.raises(ProviderResponseError):
        _provider().fetch_funding("BTC", start, end)


def test_venue_is_binance_futures(httpx_mock: HTTPXMock) -> None:
    assert _provider().venue == "binance_futures"
