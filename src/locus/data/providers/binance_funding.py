"""
binance_funding.py

Binance Futures funding rate provider.
Endpoint: GET /fapi/v1/fundingRate on fapi.binance.com.

Response shape (verified):
    [{"symbol": "BTCUSDT", "fundingTime": 1760006400000,
      "fundingRate": "0.00010000", "markPrice": "...",
      "rateType": "Regular"}]

`fundingTime` in ms (may have 3ms drift), `fundingRate` as string
(parsed to float). Pagination via `startTime` cursor, 1000 events per
request (~333 days at 8h).

Two methods:
- `fetch_funding`: DataFrame only, for callers that don't need raw.
- `fetch_funding_with_raw`: DataFrame + per-batch raw records, for
  the FundingIngestor (raw layer audit).

Author: Marco Pérez Padilla
Date:   23-10-2026
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

from locus.data.providers.funding_base import FUNDING_SCHEMA
from locus.data.symbols import SymbolMapper
from locus.data.types import ProviderResponseError
from locus.infrastructure.config import FundingConfig
from locus.infrastructure.logging import LoggerFactory


class BinanceFundingProvider:
    """Fetch 8h funding rates from Binance Futures."""

    def __init__(
        self,
        config: FundingConfig,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._client = client or httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )
        self._log = LoggerFactory.get("data.providers.binance_funding")

    @property
    def venue(self) -> str:
        return "binance_futures"

    def fetch_funding(self, asset_id: str, start: datetime, end: datetime) -> pl.DataFrame:
        symbol = SymbolMapper.to_symbol(asset_id, "binance")
        rows: list[dict] = []
        cursor_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        while cursor_ms < end_ms:
            batch, last_ms, _raw = self._fetch_batch_with_raw(symbol, cursor_ms, end_ms)
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < self._config.max_events_per_request:
                break
            if last_ms < cursor_ms:
                raise ProviderResponseError("funding provider returned non-advancing timestamps")
            cursor_ms = last_ms + 1

        if not rows:
            return pl.DataFrame(schema=FUNDING_SCHEMA)

        df = pl.DataFrame(rows, schema=FUNDING_SCHEMA).with_columns(
            pl.lit(asset_id).alias("asset_id")
        )
        return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end)).sort(
            "timestamp"
        )

    def fetch_funding_with_raw(
        self, asset_id: str, start: datetime, end: datetime
    ) -> tuple[pl.DataFrame, list[dict]]:
        """Like fetch_funding, but also returns the literal JSON records
        from every batch request, for the raw layer.

        Each raw_record dict has: symbol, request_start_ms,
        request_end_ms, response (the literal JSON list of that batch).
        """
        symbol = SymbolMapper.to_symbol(asset_id, "binance")
        rows: list[dict] = []
        raw_records: list[dict] = []
        cursor_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        while cursor_ms < end_ms:
            batch, last_ms, raw_response = self._fetch_batch_with_raw(symbol, cursor_ms, end_ms)
            if not batch:
                break
            rows.extend(batch)
            raw_records.append(
                {
                    "symbol": symbol,
                    "request_start_ms": cursor_ms,
                    "request_end_ms": end_ms,
                    "response": raw_response,
                }
            )
            if len(batch) < self._config.max_events_per_request:
                break
            if last_ms < cursor_ms:
                raise ProviderResponseError("funding provider returned non-advancing timestamps")
            cursor_ms = last_ms + 1

        if not rows:
            return pl.DataFrame(schema=FUNDING_SCHEMA), raw_records

        df = pl.DataFrame(rows, schema=FUNDING_SCHEMA).with_columns(
            pl.lit(asset_id).alias("asset_id")
        )
        df = df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end)).sort(
            "timestamp"
        )
        return df, raw_records

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _fetch_batch_with_raw(
        self, symbol: str, start_ms: int, end_ms: int
    ) -> tuple[list[dict], int, list[dict]]:
        response = self._client.get(
            "/fapi/v1/fundingRate",
            params={
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": self._config.max_events_per_request,
            },
        )
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, list):
            raise ProviderResponseError(f"unexpected funding response shape: {type(raw).__name__}")
        if not raw:
            return [], 0, []

        parsed = [
            {
                "asset_id": "",
                "venue": self.venue,
                "symbol": symbol,
                "timestamp": datetime.fromtimestamp(int(r["fundingTime"]) / 1000, tz=UTC),
                "funding_rate": float(r["fundingRate"]),
            }
            for r in raw
        ]
        last_ms = int(raw[-1]["fundingTime"])
        return parsed, last_ms, raw
