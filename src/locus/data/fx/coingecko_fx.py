"""
coingecko_fx.py

USDT/USD daily rates from CoinGecko's public API. Requires a free demo
API key (header `x-cg-demo-api-key`). Enforces the demo tier's quota
(100 calls/minute, 10,000 calls/month) client-side to avoid 429s.

Last-resort fallback for USDT/USD when Kraken fails.

Endpoint: GET /coins/tether/market_chart?vs_currency=usd&days=max&interval=daily

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from locus.data.fx.base import FX_PROVIDER_SCHEMA
from locus.data.fx.rate_limiter import CoinGeckoRateLimiter
from locus.data.fx.types import FxProviderError
from locus.infrastructure.logging import LoggerFactory


class CoinGeckoUSDTUSDProvider:
    """Fetch USDT/USD daily rates from CoinGecko."""

    def __init__(
        self,
        api_key: str,
        quota_path: Path,
        max_per_minute: int = 100,
        max_per_month: int = 10_000,
        warn_threshold: int = 9_500,
        base_url: str = "https://api.coingecko.com/api/v3",
        timeout_seconds: int = 30,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise FxProviderError(
                "CoinGecko demo API key not configured. "
                "Set DATA__FX__COINGECKO_API_KEY in .env "
                "(get one at https://www.coingecko.com/en/developers/dashboard)."
            )

        self._client: httpx.Client
        if client is None:
            self._client = httpx.Client(
                base_url=base_url,
                timeout=timeout_seconds,
            )
        else:
            self._client = client
        self._client.headers.update({"x-cg-demo-api-key": api_key})

        self._limiter = CoinGeckoRateLimiter(
            quota_path=quota_path,
            max_per_minute=max_per_minute,
            max_per_month=max_per_month,
            warn_threshold=warn_threshold,
        )
        self._log = LoggerFactory.get("data.fx.coingecko")

    @property
    def source(self) -> str:
        return "coingecko"

    def fetch_rates(self, pair: str, start: datetime, end: datetime) -> pl.DataFrame:
        """Fetch USDT/USD rates in [start, end) via the range endpoint.

        Uses `/market_chart/range` with explicit `from`/`to` unix
        timestamps rather than `/market_chart` with `days=max` (which
        exceeds the demo tier's daily quota). The demo tier allows up to
        ~365 days per request; larger ranges raise FxProviderError.
        """
        if pair != "USDTUSD":
            raise FxProviderError(f"coingecko fx only supports USDTUSD, got {pair}")

        days = (end - start).days
        if days > 365:
            raise FxProviderError(
                f"CoinGecko demo tier supports up to 365 days per request, "
                f"got {days}. Use Kraken for longer ranges."
            )

        raw = self._fetch(
            "/coins/tether/market_chart/range",
            params={
                "vs_currency": "usd",
                "from": int(start.timestamp()),
                "to": int(end.timestamp()),
            },
        )

        prices = raw.get("prices", [])
        if not isinstance(prices, list):
            raise FxProviderError(f"unexpected response shape: prices is {type(prices).__name__}")

        by_day: dict[datetime, float] = {}
        for ts_ms, price in prices:
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=UTC).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            if not (start.date() <= ts.date() < end.date()):
                continue
            by_day[ts] = float(price)

        if not by_day:
            return pl.DataFrame(schema=FX_PROVIDER_SCHEMA)

        rows = [{"pair": "USDTUSD", "date": ts, "rate": rate} for ts, rate in by_day.items()]
        return pl.DataFrame(rows, schema=FX_PROVIDER_SCHEMA).sort("date")

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _fetch(self, path: str, params: dict) -> dict:
        self._limiter.wait_and_reserve()
        response = self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()
