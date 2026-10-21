"""
kraken_fx.py

USDT/USD daily rates from Kraken's public OHLC endpoint. Reuses the
Kraken public API (`/0/public/OHLC?pair=USDTUSD&interval=1440`) but
extracts the close price only.

Kraken's public API returns only the most recent 720 bars (720 days for
daily interval). Historical ranges older than ~2 years are not
available; use CoinGecko as fallback for older data.

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

from locus.data.fx.base import FX_PROVIDER_SCHEMA
from locus.data.fx.types import FxProviderError
from locus.infrastructure.logging import LoggerFactory

_KRAKEN_FX_PAIR = {
    "USDTUSD": "USDTUSD",
}


class KrakenUSDTUSDProvider:
    """Fetch USDT/USD daily rates from Kraken."""

    def __init__(
        self,
        base_url: str = "https://api.kraken.com",
        timeout_seconds: int = 30,
        client: httpx.Client | None = None,
    ) -> None:
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_seconds)
        self._log = LoggerFactory.get("data.fx.kraken")

    @property
    def source(self) -> str:
        return "kraken"

    def fetch_rates(self, pair: str, start: datetime, end: datetime) -> pl.DataFrame:
        """Fetch USDT/USD rates. Raises FxProviderError for other pairs."""
        if pair not in _KRAKEN_FX_PAIR:
            raise FxProviderError(f"kraken fx only supports {list(_KRAKEN_FX_PAIR)}, got {pair}")
        params = {
            "pair": _KRAKEN_FX_PAIR[pair],
            "interval": 1440,
            "since": int(start.timestamp()),
        }
        raw = self._fetch("/0/public/OHLC", params)

        errors = raw.get("error", [])
        if errors:
            raise FxProviderError(f"kraken error: {errors}")

        result = raw.get("result", {})
        bars_key = next(
            (k for k, v in result.items() if k != "last" and isinstance(v, list)),
            None,
        )
        if bars_key is None:
            return pl.DataFrame(schema=FX_PROVIDER_SCHEMA)

        rows = []
        for bar in result[bars_key]:
            ts = datetime.fromtimestamp(int(bar[0]), tz=UTC)
            if not (start <= ts < end):
                continue
            rows.append({"pair": pair, "date": ts, "rate": float(bar[4])})
        if not rows:
            return pl.DataFrame(schema=FX_PROVIDER_SCHEMA)

        return pl.DataFrame(rows, schema=FX_PROVIDER_SCHEMA).sort("date")

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _fetch(self, path: str, params: dict) -> dict:
        response = self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()
