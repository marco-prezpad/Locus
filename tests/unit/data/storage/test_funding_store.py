"""
test_funding_store.py

FundingStore processed + raw roundtrip.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from locus.data.providers.funding_base import FUNDING_SCHEMA
from locus.data.storage.funding_store import FundingStore


def _df() -> pl.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        {
            "asset_id": "BTC",
            "venue": "binance_futures",
            "symbol": "BTCUSDT",
            "timestamp": base + timedelta(hours=8 * i),
            "funding_rate": 0.0001 + 0.00001 * i,
        }
        for i in range(3)
    ]
    return pl.DataFrame(rows, schema=FUNDING_SCHEMA)


def test_write_and_read_processed(tmp_path: Path) -> None:
    store = FundingStore(tmp_path / "processed", tmp_path / "raw")
    written = store.write(_df())
    assert written == 3
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 10, tzinfo=UTC)
    df = store.read("binance_futures", "BTC", start, end)
    assert df.height == 3


def test_raw_append(tmp_path: Path) -> None:
    store = FundingStore(tmp_path / "processed", tmp_path / "raw")
    store.write_raw(
        venue="binance_futures",
        asset_id="BTC",
        symbol="BTCUSDT",
        request_start_ms=1000,
        request_end_ms=2000,
        response_json='[{"symbol":"BTCUSDT"}]',
    )
    store.write_raw(
        venue="binance_futures",
        asset_id="BTC",
        symbol="BTCUSDT",
        request_start_ms=2000,
        request_end_ms=3000,
        response_json='[{"symbol":"BTCUSDT"}]',
    )
    raw = store.read_raw("binance_futures", "BTC")
    assert raw.height == 2


def test_raw_empty_when_missing(tmp_path: Path) -> None:
    store = FundingStore(tmp_path / "processed", tmp_path / "raw")
    raw = store.read_raw("binance_futures", "BTC")
    assert raw.height == 0
