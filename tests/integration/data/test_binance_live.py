"""
test_binance_live.py

Integration tests that hit the real Binance API. Skipped by default;
run with `pytest -m integration tests/integration`.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from locus.data.providers.binance import BinanceMarketDataProvider
from locus.data.types import BarRequest, Timeframe
from locus.data.validation.ohlcv import validate_ohlcv
from locus.infrastructure.config import BinanceProviderConfig

pytestmark = pytest.mark.integration


def test_fetch_real_btcusdt_one_week() -> None:
    provider = BinanceMarketDataProvider(BinanceProviderConfig())
    df = provider.fetch_bars(
        BarRequest(
            asset_id="BTC",
            timeframe=Timeframe.H1,
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=datetime(2025, 1, 8, tzinfo=UTC),
        )
    )
    # 168 hours in a week; allow up to 2 missing bars (exchange gaps)
    assert df.height >= 166
    validate_ohlcv(df)
