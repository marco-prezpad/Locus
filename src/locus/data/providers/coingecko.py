"""
coingecko.py

CoinGecko public API provider. Last-resort fallback: only daily bars
are supported, and volume is reported as 0 because the /ohlc endpoint
does not expose volume. Rate limits are tight (typically 10-30 req/min
on the free tier); the provider is meant to be used only when Binance
and Kraken both fail.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import polars as pl
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from locus.data.providers.base import BAR_SCHEMA
from locus.data.symbols import SymbolMapper
from locus.data.types import (
    BarRequest,
    ProviderResponseError,
    Timeframe,
)
from locus.infrastructure.config import CoinGeckoProviderConfig
from locus.infrastructure.logging import LoggerFactory


class CoinGeckoMarketDataProvider:
    """Fetch daily OHLC bars from CoinGecko's public API."""

    def __init__(
        self,
        config: CoinGeckoProviderConfig,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._client = client or httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )
        self._log = LoggerFactory.get("data.providers.coingecko")

    @property
    def venue(self) -> str:
        return "coingecko"

    def fetch_bars(self, request: BarRequest) -> pl.DataFrame:
        """Fetch bars via /coins/{id}/ohlc.

        Only Timeframe.D1 is supported. Intraday timeframes raise
        ProviderResponseError: CoinGecko's free tier returns granularity
        that does not match any of our canonical intraday intervals.
        """
        if request.timeframe != Timeframe.D1:
            raise ProviderResponseError(
                f"coingecko only supports 1d timeframe, got {request.timeframe}"
            )
        coin_id = SymbolMapper.to_symbol(request.asset_id, self.venue)
        days = max((request.end - request.start).days + 1, 1)
        rows = self._fetch_ohlc(
            coin_id=coin_id, vs_currency=self._config.default_vs_currency, days=days
        )
        if not rows:
            return pl.DataFrame(schema=BAR_SCHEMA)

        df = pl.DataFrame(rows, schema=BAR_SCHEMA)
        return df.filter(
            (pl.col("timestamp") >= request.start) & (pl.col("timestamp") < request.end)
        ).sort("timestamp")

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def _fetch_ohlc(self, coin_id: str, vs_currency: str, days: int) -> list[dict[str, object]]:
        url = f"/coins/{coin_id}/ohlc"
        params = {"vs_currency": vs_currency, "days": days}
        response = self._client.get(url, params=params)
        response.raise_for_status()
        raw = response.json()

        if not isinstance(raw, list):
            raise ProviderResponseError(f"unexpected response shape: {type(raw).__name__}")

        return [
            {
                "asset_id": "",
                "venue": self.venue,
                "symbol": coin_id,
                "timeframe": "1d",
                "timestamp": datetime.fromtimestamp(k[0] / 1000, tz=UTC),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": 0.0,
            }
            for k in raw
        ]
