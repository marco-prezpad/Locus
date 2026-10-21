"""
binance.py

Binance public market data provider. Uses the spot REST API
(/api/v3/klines). No API key required for historical OHLCV.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import polars as pl
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from locus.data.providers.base import BAR_SCHEMA
from locus.data.types import (
    BarRequest,
    ProviderNoDataError,
    ProviderResponseError,
    Timeframe,
)
from locus.infrastructure.config import BinanceProviderConfig
from locus.infrastructure.logging import LoggerFactory

_BINANCE_TIMEFRAME: dict[Timeframe, str] = {
    Timeframe.M1: "1m",
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1d",
}

_TIMEFRAME_DELTA: dict[Timeframe, timedelta] = {
    Timeframe.M1: timedelta(minutes=1),
    Timeframe.M5: timedelta(minutes=5),
    Timeframe.M15: timedelta(minutes=15),
    Timeframe.H1: timedelta(hours=1),
    Timeframe.H4: timedelta(hours=4),
    Timeframe.D1: timedelta(days=1),
}


class BinanceMarketDataProvider:
    """Fetch OHLCV bars from Binance's public spot API."""

    def __init__(
        self,
        config: BinanceProviderConfig,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._client = client or httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )
        self._log = LoggerFactory.get("data.providers.binance")

    @property
    def venue(self) -> str:
        return "binance"

    def fetch_bars(self, request: BarRequest) -> pl.DataFrame:
        """Fetch bars by paginating through /api/v3/klines.

        Binance caps each response at `max_bars_per_request` (1000 by
        default). This method advances the cursor by the last open time
        of each batch until the requested end is reached.
        """
        if request.timeframe not in _BINANCE_TIMEFRAME:
            raise ProviderResponseError(f"unsupported timeframe: {request.timeframe}")

        interval = _BINANCE_TIMEFRAME[request.timeframe]
        cursor = request.start
        rows: list[dict[str, object]] = []

        while cursor < request.end:
            batch = self._fetch_batch(
                symbol=request.symbol,
                interval=interval,
                start=cursor,
                end=request.end,
            )
            if not batch:
                break
            rows.extend(batch)
            last_open = batch[-1]["timestamp"]
            assert isinstance(last_open, datetime)
            if last_open < cursor:
                raise ProviderResponseError("provider returned non-advancing timestamps")
            cursor = last_open + _TIMEFRAME_DELTA[request.timeframe]

        if not rows:
            return pl.DataFrame(schema=BAR_SCHEMA)

        df = pl.DataFrame(rows, schema=BAR_SCHEMA)
        return df.filter(
            (pl.col("timestamp") >= request.start) & (pl.col("timestamp") < request.end)
        ).sort("timestamp")

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, ProviderNoDataError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _fetch_batch(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, object]]:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "limit": self._config.max_bars_per_request,
        }
        response = self._client.get("/api/v3/klines", params=params)
        response.raise_for_status()
        raw = response.json()

        if not isinstance(raw, list):
            raise ProviderResponseError(f"unexpected response shape: {type(raw).__name__}")

        return [
            {
                "asset_id": "",
                "venue": self.venue,
                "symbol": symbol,
                "timeframe": interval,
                "timestamp": datetime.fromtimestamp(k[0] / 1000, tz=UTC),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
            for k in raw
        ]
