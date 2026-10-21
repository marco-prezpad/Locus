"""
test_coingecko_fx.py

CoinGeckoUSDTUSDProvider: API key header, range endpoint, rate limiter.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pytest_httpx import HTTPXMock

from locus.data.fx.coingecko_fx import CoinGeckoUSDTUSDProvider
from locus.data.fx.types import FxProviderError


def _provider(tmp_path: Path, api_key: str = "test-key") -> CoinGeckoUSDTUSDProvider:
    return CoinGeckoUSDTUSDProvider(
        api_key=api_key,
        quota_path=tmp_path / "quota.json",
        base_url="https://api.coingecko.com/api/v3",
        client=httpx.Client(base_url="https://api.coingecko.com/api/v3"),
    )


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def test_rates_returned_from_range(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 4, tzinfo=UTC)
    from_ts = int(start.timestamp())
    to_ts = int(end.timestamp())
    httpx_mock.add_response(
        url=(
            "https://api.coingecko.com/api/v3/coins/tether/market_chart/range"
            f"?vs_currency=usd&from={from_ts}&to={to_ts}"
        ),
        json={
            "prices": [
                [_ms(start), 0.998],
                [_ms(datetime(2024, 1, 2, tzinfo=UTC)), 0.999],
                [_ms(datetime(2024, 1, 3, tzinfo=UTC)), 1.001],
            ]
        },
        match_headers={"x-cg-demo-api-key": "test-key"},
    )
    df = _provider(tmp_path).fetch_rates("USDTUSD", start, end)
    assert df.height == 3
    assert df["rate"].to_list() == [0.998, 0.999, 1.001]


def test_api_key_sent_in_header(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    from_ts = int(start.timestamp())
    to_ts = int(end.timestamp())
    httpx_mock.add_response(
        url=(
            "https://api.coingecko.com/api/v3/coins/tether/market_chart/range"
            f"?vs_currency=usd&from={from_ts}&to={to_ts}"
        ),
        json={"prices": [[_ms(start), 0.999]]},
        match_headers={"x-cg-demo-api-key": "test-key"},
    )
    df = _provider(tmp_path).fetch_rates("USDTUSD", start, end)
    assert df.height == 1


def test_missing_api_key_raises(tmp_path: Path) -> None:
    with pytest.raises(FxProviderError, match="API key not configured"):
        CoinGeckoUSDTUSDProvider(
            api_key="",
            quota_path=tmp_path / "quota.json",
        )


def test_unsupported_pair_raises(tmp_path: Path) -> None:
    with pytest.raises(FxProviderError):
        _provider(tmp_path).fetch_rates(
            "EURUSD",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
        )


def test_range_exceeds_demo_tier_raises(tmp_path: Path) -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(FxProviderError, match="up to 365 days"):
        _provider(tmp_path).fetch_rates("USDTUSD", start, end)


def test_source_is_coingecko(tmp_path: Path) -> None:
    assert _provider(tmp_path).source == "coingecko"
