"""
discovery.py

Discovers asset listings from Binance's public exchangeInfo endpoint.
Listing dates come from `onboardDate` when present, and fall back to the
timestamp of the earliest available 1d bar otherwise.

Kraken and CoinGecko do not expose listing dates via public APIs;
discovery is Binance-only in the MVP. Documented in `docs/data.md`.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from locus.data.symbols import SymbolMapper
from locus.data.types import ProviderResponseError
from locus.data.universe.types import AssetListing
from locus.infrastructure.config import BinanceProviderConfig
from locus.infrastructure.logging import LoggerFactory


class BinanceUniverseDiscovery:
    """Discover listing dates from Binance exchangeInfo."""

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
        self._log = LoggerFactory.get("data.universe.discovery.binance")

    def discover(self, asset_ids: list[str]) -> list[AssetListing]:
        """Return an AssetListing per (asset, "binance") for every asset
        in `asset_ids` that appears in exchangeInfo.

        Listing date comes from `onboardDate` when present in the
        exchangeInfo response, and falls back to the timestamp of the
        earliest available 1d bar otherwise. Assets that the
        SymbolMapper does not know for "binance", or that Binance does
        not list (yet or ever), are silently skipped.
        """
        symbols_to_asset = {
            SymbolMapper.to_symbol(asset_id, "binance"): asset_id for asset_id in asset_ids
        }

        raw = self._fetch_exchange_info()
        onboard_by_symbol: dict[str, int] = {
            entry["symbol"]: entry["onboardDate"]
            for entry in raw
            if entry.get("status") == "TRADING"
            and entry.get("symbol") in symbols_to_asset
            and "onboardDate" in entry
        }

        listings: list[AssetListing] = []
        for symbol, asset_id in symbols_to_asset.items():
            onboard_ms = onboard_by_symbol.get(symbol)
            if onboard_ms is not None:
                first_ts = datetime.fromtimestamp(onboard_ms / 1000, tz=UTC)
            else:
                first_ts = self._fetch_first_bar_ts(symbol)
                if first_ts is None:
                    self._log.warning(
                        "asset_not_listed",
                        extra={"asset": asset_id, "venue": "binance"},
                    )
                    continue

            listings.append(
                AssetListing(
                    asset_id=asset_id,
                    venue="binance",
                    first_tradable_ts=first_ts,
                )
            )
        return listings

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _fetch_exchange_info(self) -> list[dict]:
        """Fetch and return the raw `symbols` array from exchangeInfo.

        Raises `ProviderResponseError` if the response shape is not as
        expected, so callers can distinguish parse failures from
        transport errors.
        """
        response = self._client.get("/api/v3/exchangeInfo")
        response.raise_for_status()
        payload = response.json()
        symbols = payload.get("symbols")
        if not isinstance(symbols, list):
            raise ProviderResponseError("exchangeInfo response missing 'symbols' list")
        return symbols

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _fetch_first_bar_ts(self, symbol: str) -> datetime | None:
        """Return the timestamp of the earliest available 1d bar for a
        symbol, or None if the symbol has no available data.

        Used as a fallback for listing date when exchangeInfo does not
        include `onboardDate`. Binance returns the earliest available
        bar when `startTime=0` is requested.
        """
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": 0,
            "limit": 1,
        }
        response = self._client.get("/api/v3/klines", params=params)
        response.raise_for_status()
        raw = response.json()
        if not raw:
            return None
        return datetime.fromtimestamp(raw[0][0] / 1000, tz=UTC)
