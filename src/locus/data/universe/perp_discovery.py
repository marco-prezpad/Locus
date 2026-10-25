"""
perp_discovery.py

Discovery of perpetual listings from Binance Futures
`/fapi/v1/exchangeInfo`.

The exchangeInfo response contains crypto perps (contractType=PERPETUAL,
underlyingType=COIN), equity perps (TRADIFI_PERPETUAL,
underlyingType=EQUITY/HK_EQUITY/KR_EQUITY/CN_EQUITY/FX), quarterly
futures (NEXT_QUARTER), and premarket (PREMARKET). Two filters are
required:
- contractType == "PERPETUAL" (excludes all the rest)
- underlyingType == "COIN" (double safeguard)
Without both, `BTCUSDT` (crypto perp) would collide with
`BTCUSDT_270326` (quarterly) in the symbol map.

Unlike spot, `onboardDate` IS present in perp exchangeInfo. No
first-bar fallback needed.

Author: Marco Pérez Padilla
Date:   23-10-2026
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
from locus.data.universe.perp_types import PerpListing
from locus.infrastructure.config import FundingConfig
from locus.infrastructure.logging import LoggerFactory


class PerpUniverseDiscovery:
    """Discover perp listings from Binance Futures exchangeInfo."""

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
        self._log = LoggerFactory.get("data.universe.perp_discovery")

    def discover(self, asset_ids: list[str]) -> list[PerpListing]:
        symbols_to_asset = {
            SymbolMapper.to_symbol(asset_id, "binance"): asset_id for asset_id in asset_ids
        }
        raw = self._fetch_exchange_info()

        listings: list[PerpListing] = []
        for entry in raw:
            symbol = entry.get("symbol", "")
            if symbol not in symbols_to_asset:
                continue
            if entry.get("contractType") != "PERPETUAL":
                continue
            if entry.get("underlyingType") != "COIN":
                continue
            if entry.get("status") != "TRADING":
                continue
            onboard_ms = entry.get("onboardDate")
            if onboard_ms is None:
                self._log.warning(
                    "perp_no_onboard_date",
                    extra={"symbol": symbol},
                )
                continue
            listings.append(
                PerpListing(
                    asset_id=symbols_to_asset[symbol],
                    venue="binance_futures",
                    list_date=datetime.fromtimestamp(onboard_ms / 1000, tz=UTC),
                    delist_date=None,
                )
            )
        self._log.info(
            "perp_discovery_done",
            extra={
                "requested": len(asset_ids),
                "discovered": len(listings),
            },
        )
        return listings

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _fetch_exchange_info(self) -> list[dict]:
        response = self._client.get("/fapi/v1/exchangeInfo")
        response.raise_for_status()
        payload = response.json()
        symbols = payload.get("symbols")
        if not isinstance(symbols, list):
            raise ProviderResponseError("perp exchangeInfo missing 'symbols' list")
        return symbols
