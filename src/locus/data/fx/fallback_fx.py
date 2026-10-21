"""
fallback_fx.py

FallbackFxProvider tries providers in order and returns the first
non-empty DataFrame. Unlike the market-data fallback, it does not
merge results across providers: FX rates from different sources may
disagree, and mixing them silently would corrupt downstream PnL. The
caller sees one source's data, tagged with that source's name.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from locus.data.fx.base import FxRateProvider
from locus.data.fx.types import FxError, FxProviderError
from locus.infrastructure.logging import LoggerFactory


class FallbackFxProvider:
    """Try each FX provider in order until one returns non-empty data."""

    def __init__(self, providers: list[FxRateProvider]) -> None:
        if not providers:
            raise ValueError("providers list must be non-empty")
        self._providers = providers
        self._log = LoggerFactory.get("data.fx.fallback")

    @property
    def source(self) -> str:
        return "fallback"

    def fetch_rates(self, pair: str, start: datetime, end: datetime) -> pl.DataFrame:
        """Try providers in order; return the first non-empty result.

        Raises the last ProviderError if every provider fails.
        """
        last_error: FxError | None = None
        for provider in self._providers:
            try:
                df = provider.fetch_rates(pair, start, end)
            except FxProviderError as exc:
                self._log.warning(
                    "fx_provider_failed",
                    extra={"pair": pair, "source": provider.source, "error": str(exc)},
                )
                last_error = exc
                continue

            if df.height > 0:
                self._log.info(
                    "fx_provider_used",
                    extra={"pair": pair, "source": provider.source},
                )
                return df

            self._log.info(
                "fx_provider_empty",
                extra={"pair": pair, "source": provider.source},
            )

        if last_error is not None:
            raise last_error

        from locus.data.fx.base import FX_PROVIDER_SCHEMA

        return pl.DataFrame(schema=FX_PROVIDER_SCHEMA)
