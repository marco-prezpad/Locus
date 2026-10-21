"""
test_kraken_live.py

Integration tests against the real Kraken public API. Runs in CI
(Kraken does not block datacenter IPs). Kraken only returns recent
data (720-bar cap), so we request a window from the last 2 days.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from locus.data.providers.kraken import KrakenMarketDataProvider
from locus.data.types import BarRequest, Timeframe
from locus.data.validation.ohlcv import validate_ohlcv
from locus.infrastructure.config import KrakenProviderConfig

pytestmark = pytest.mark.integration


def test_fetch_real_btcusdt_recent() -> None:
    provider = KrakenMarketDataProvider(KrakenProviderConfig())
    end = datetime.now(UTC)
    start = end - timedelta(days=2)
    df = provider.fetch_bars(
        BarRequest(
            asset_id="BTC",
            timeframe=Timeframe.H1,
            start=start,
            end=end,
        )
    )
    # 48 hours requested; Kraken may return a handful fewer due to
    # boundaries. Accept at least 40.
    assert df.height >= 40
    validate_ohlcv(df)
