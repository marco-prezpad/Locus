"""
test_frankfurter.py

FrankfurterEURUSDProvider with mocked HTTP.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from pytest_httpx import HTTPXMock

from locus.data.fx.frankfurter import FrankfurterEURUSDProvider
from locus.data.fx.types import FxProviderError


def _provider() -> FrankfurterEURUSDProvider:
    return FrankfurterEURUSDProvider(
        base_url="https://api.frankfurter.app",
        client=httpx.Client(base_url="https://api.frankfurter.app"),
    )


def test_eurusd_returns_rates(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=("https://api.frankfurter.app/2024-01-01..2024-01-03?from=EUR&to=USD"),
        json={
            "amount": 1.0,
            "base": "EUR",
            "start_date": "2024-01-01",
            "end_date": "2024-01-03",
            "rates": {
                "2024-01-02": {"USD": 1.10},
                "2024-01-03": {"USD": 1.11},
            },
        },
    )
    df = _provider().fetch_rates(
        "EURUSD",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 4, tzinfo=UTC),
    )
    assert df.height == 2
    assert df["rate"].to_list() == [1.10, 1.11]


def test_unsupported_pair_raises() -> None:
    with pytest.raises(FxProviderError):
        _provider().fetch_rates(
            "GBPUSD",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
        )


def test_empty_rates_returns_empty(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=("https://api.frankfurter.app/2024-01-01..2024-01-02?from=EUR&to=USD"),
        json={"rates": {}},
    )
    df = _provider().fetch_rates(
        "EURUSD",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
    )
    assert df.height == 0


def test_source_is_frankfurter() -> None:
    assert _provider().source == "frankfurter"
