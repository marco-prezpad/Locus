"""
kraken.py

Kraken public market data provider. Uses /0/public/OHLC. No API key
required. Pagination is driven by the `since` cursor that Kraken
returns in each response.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

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
    ProviderNoDataError,
    ProviderResponseError,
    Timeframe,
)
from locus.infrastructure.config import KrakenProviderConfig
from locus.infrastructure.logging import LoggerFactory

_KRAKEN_INTERVAL: dict[Timeframe, int] = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
}

_INTERVAL_DELTA: dict[Timeframe, timedelta] = {
    Timeframe.M1: timedelta(minutes=1),
    Timeframe.M5: timedelta(minutes=5),
    Timeframe.M15: timedelta(minutes=15),
    Timeframe.H1: timedelta(hours=1),
    Timeframe.H4: timedelta(hours=4),
    Timeframe.D1: timedelta(days=1),
}


class KrakenMarketDataProvider:
    """Fetch OHLCV bars from Kraken's public spot API."""

    def __init__(
        self,
        config: KrakenProviderConfig,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._client = client or httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )
        self._log = LoggerFactory.get("data.providers.kraken")

    @property
    def venue(self) -> str:
        return "kraken"

    def fetch_bars(self, request: BarRequest) -> pl.DataFrame:
        """Fetch bars by paginating with Kraken's `since` cursor.

        Kraken caps each response at `max_bars_per_request` (720 by
        default). The cursor advances past the last returned bar to avoid
        duplicates across pages.
        """
        if request.timeframe not in _KRAKEN_INTERVAL:
            raise ProviderResponseError(f"unsupported timeframe: {request.timeframe}")

        interval = _KRAKEN_INTERVAL[request.timeframe]
        delta = _INTERVAL_DELTA[request.timeframe]
        cursor = request.start
        rows: list[dict[str, object]] = []

        while cursor < request.end:
            batch, last_ts = self._fetch_batch(
                symbol=SymbolMapper.to_symbol(request.asset_id, self.venue),
                interval=interval,
                since=cursor,
            )
            if not batch:
                break
            rows.extend(batch)
            if last_ts <= cursor:
                raise ProviderResponseError("provider returned non-advancing timestamps")
            cursor = last_ts + delta

        if not rows:
            return pl.DataFrame(schema=BAR_SCHEMA)

        df = pl.DataFrame(rows, schema=BAR_SCHEMA)
        if rows:
            timestamps = [cast(datetime, r["timestamp"]) for r in rows]
            first_ts = min(timestamps)
            if first_ts > request.end:
                raise ProviderNoDataError(
                    f"kraken returned data starting at {first_ts}, "
                    f"after requested end {request.end} "
                    f"(public OHLC API returns only the most recent 720 bars)"
                )
        return df.filter(
            (pl.col("timestamp") >= request.start) & (pl.col("timestamp") < request.end)
        ).sort("timestamp")

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _fetch_batch(
        self,
        symbol: str,
        interval: int,
        since: datetime,
    ) -> tuple[list[dict[str, object]], datetime]:
        params = {
            "pair": symbol,
            "interval": interval,
            "since": int(since.timestamp()),
        }
        response = self._client.get("/0/public/OHLC", params=params)
        response.raise_for_status()
        raw = response.json()

        errors = raw.get("error", [])
        if errors:
            raise ProviderResponseError(f"kraken error: {errors}")

        result = raw.get("result", {})
        last_ts_unix = result.get("last")
        if last_ts_unix is None:
            raise ProviderResponseError("kraken response missing 'last'")

        bars_key = next(
            (k for k, v in result.items() if k != "last" and isinstance(v, list)),
            None,
        )
        if bars_key is None:
            return [], datetime.fromtimestamp(last_ts_unix, tz=UTC)

        parsed = [
            {
                "asset_id": "",
                "venue": self.venue,
                "symbol": symbol,
                "timeframe": str(interval),
                "timestamp": datetime.fromtimestamp(int(k[0]), tz=UTC),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[6]),
            }
            for k in result[bars_key]
        ]
        last_ts = (
            parsed[-1]["timestamp"] if parsed else datetime.fromtimestamp(last_ts_unix, tz=UTC)
        )
        assert isinstance(last_ts, datetime)
        return parsed, last_ts
