"""
test_kraken_fx.py

KrakenUSDTUSDProvider with mocked HTTP.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from pytest_httpx import HTTPXMock

from locus.data.fx.kraken_fx import KrakenUSDTUSDProvider
from locus.data.fx.types import FxProviderError


def _provider() -> KrakenUSDTUSDProvider:
    return KrakenUSDTUSDProvider(
        base_url="https://api.kraken.com",
        client=httpx.Client(base_url="https://api.kraken.com"),
    )


def _kraken_bar(unix_ts: int, close: float) -> list:
    return [
        unix_ts,
        str(close - 0.001),
        str(close + 0.001),
        str(close - 0.002),
        str(close),
        "1000.0",
        str(close),
    ]


def test_rates_returned(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        json={
            "error": [],
            "result": {
                "USDTUSD": [
                    _kraken_bar(1704153600, 0.999),  # 2024-01-02
                    _kraken_bar(1704240000, 1.000),  # 2024-01-03
                ],
                "last": 1704240000,
            },
        }
    )
    df = _provider().fetch_rates(
        "USDTUSD",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 4, tzinfo=UTC),
    )
    assert df.height == 2
    assert df["rate"].to_list() == [0.999, 1.000]


def test_unsupported_pair_raises() -> None:
    with pytest.raises(FxProviderError):
        _provider().fetch_rates(
            "EURUSD",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
        )


def test_kraken_error_raises(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(json={"error": ["EQuery:Unknown asset pair"], "result": {}})
    with pytest.raises(FxProviderError):
        _provider().fetch_rates(
            "USDTUSD",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
        )


def test_source_is_kraken() -> None:
    assert _provider().source == "kraken"
