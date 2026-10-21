"""
test_binance_live.py

Integration tests that hit the real Binance API. Skipped by default;
run with `pytest -m integration tests/integration`.

Binance returns `451 Unavailable For Legal Reasons` for requests
originating from datacenter IPs (including GitHub Actions runners in
AWS). The test detects this and skips with a clear reason instead of
failing. Run locally from a residential IP to actually exercise it.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from locus.data.providers.binance import BinanceMarketDataProvider
from locus.data.types import BarRequest, Timeframe
from locus.data.validation.ohlcv import validate_ohlcv
from locus.infrastructure.config import BinanceProviderConfig

pytestmark = pytest.mark.integration


def test_fetch_real_btcusdt_one_week() -> None:
    provider = BinanceMarketDataProvider(BinanceProviderConfig())
    try:
        df = provider.fetch_bars(
            BarRequest(
                asset_id="BTC",
                timeframe=Timeframe.H1,
                start=datetime(2025, 1, 1, tzinfo=UTC),
                end=datetime(2025, 1, 8, tzinfo=UTC),
            )
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 451:
            pytest.skip(
                "Binance blocks datacenter IPs (451 Unavailable For Legal "
                "Reasons); run locally from a residential IP to exercise "
                "this test."
            )
        raise

    # 168 hours in a week; allow up to 2 missing bars (exchange gaps)
    assert df.height >= 166
    validate_ohlcv(df)
