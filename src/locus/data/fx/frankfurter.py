"""
frankfurter.py

EUR/USD daily rates from the Frankfurter API (ECB data). Free, no API
key, full history from 1999. Only EUR-base pairs are available; the
provider is dedicated to EUR/USD in the MVP.

Endpoint: https://api.frankfurter.app/{start}..{end}?from=EUR&to=USD
Response: {"rates": {"YYYY-MM-DD": {"USD": 1.10}, ...}}

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

from locus.data.fx.base import FX_PROVIDER_SCHEMA
from locus.data.fx.types import FxProviderError
from locus.infrastructure.logging import LoggerFactory


class FrankfurterEURUSDProvider:
    """Fetch EUR/USD daily rates from Frankfurter (ECB)."""

    def __init__(
        self,
        base_url: str = "https://api.frankfurter.dev/v1",
        timeout_seconds: int = 30,
        client: httpx.Client | None = None,
    ) -> None:
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_seconds)
        self._log = LoggerFactory.get("data.fx.frankfurter")

    @property
    def source(self) -> str:
        return "frankfurter"

    def fetch_rates(self, pair: str, start: datetime, end: datetime) -> pl.DataFrame:
        """Fetch EUR/USD rates in the half-open interval [start, end).

        Frankfurter's range API is inclusive on both ends, so the
        requested inclusive end is `end - 1 day`. Returns an empty
        DataFrame if the range collapses (start == end or end is before
        start). Rows are additionally filtered to `[start, end)` so any
        server-side inclusive-end surprises do not leak.
        """
        if pair != "EURUSD":
            raise FxProviderError(f"frankfurter only supports EURUSD in this provider, got {pair}")

        start_date = start.date()
        end_date_inclusive = (end - timedelta(days=1)).date()
        if end_date_inclusive < start_date:
            return pl.DataFrame(schema=FX_PROVIDER_SCHEMA)

        raw = self._fetch(
            f"/{start_date.isoformat()}..{end_date_inclusive.isoformat()}",
            params={"from": "EUR", "to": "USD"},
        )

        rates = raw.get("rates", {})
        if not isinstance(rates, dict):
            raise FxProviderError(f"unexpected response shape: rates is {type(rates).__name__}")

        rows = []
        for date_str, day_rates in rates.items():
            usd = day_rates.get("USD")
            if usd is None:
                continue
            dt = datetime.fromisoformat(date_str).replace(tzinfo=UTC)
            if not (start <= dt < end):
                continue
            rows.append({"pair": "EURUSD", "date": dt, "rate": float(usd)})
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
