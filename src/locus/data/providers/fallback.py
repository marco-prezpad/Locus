"""
fallback.py

FallbackMarketDataProvider tries a list of providers in order and
returns the first non-empty DataFrame. If all fail (error or empty),
the last error is re-raised. Designed for the chain
[Binance, Kraken, CoinGecko], but the class is provider-agnostic.

The DataFrame returned always carries the venue of the provider that
actually responded (the inner provider sets it). The `.venue` property
on this class reports "fallback" for logging purposes only.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import polars as pl

from locus.data.providers.base import MarketDataProvider
from locus.data.types import BarRequest, ProviderError
from locus.infrastructure.logging import LoggerFactory


class FallbackMarketDataProvider:
    """Try each provider in order until one returns non-empty data."""

    def __init__(self, providers: list[MarketDataProvider]) -> None:
        if not providers:
            raise ValueError("providers list must be non-empty")
        self._providers = providers
        self._log = LoggerFactory.get("data.providers.fallback")

    @property
    def venue(self) -> str:
        return "fallback"

    def fetch_bars(self, request: BarRequest) -> pl.DataFrame:
        """Try providers in order; return the first non-empty result.

        Raises the last ProviderError if every provider fails. If all
        providers succeed but return empty, returns an empty DataFrame
        with the canonical schema.
        """
        last_error: ProviderError | None = None
        for provider in self._providers:
            try:
                df = provider.fetch_bars(request)
            except ProviderError as exc:
                self._log.warning(
                    "provider_failed",
                    extra={
                        "asset": request.asset_id,
                        "venue": provider.venue,
                        "error": str(exc),
                    },
                )
                last_error = exc
                continue

            if df.height > 0:
                self._log.info(
                    "provider_used",
                    extra={"asset": request.asset_id, "venue": provider.venue},
                )
                return df

            self._log.info(
                "provider_empty",
                extra={"asset": request.asset_id, "venue": provider.venue},
            )

        if last_error is not None:
            raise last_error

        from locus.data.providers.base import BAR_SCHEMA

        return pl.DataFrame(schema=BAR_SCHEMA)
