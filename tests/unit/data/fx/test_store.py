"""
test_store.py

FxRateStore write/read round-trip.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from locus.data.fx.base import FX_PROVIDER_SCHEMA
from locus.data.fx.store import FxRateStore


def _df() -> pl.DataFrame:
    rows = [
        {
            "pair": "EURUSD",
            "date": datetime(2024, 1, 1 + i, tzinfo=UTC),
            "rate": 1.10 + 0.01 * i,
        }
        for i in range(3)
    ]
    return pl.DataFrame(rows, schema=FX_PROVIDER_SCHEMA)


def test_write_and_read(tmp_path: Path) -> None:
    store = FxRateStore(tmp_path)
    written = store.write(_df(), source="frankfurter")
    assert written == 3
    df = store.read(
        "EURUSD",
        "frankfurter",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 5, tzinfo=UTC),
    )
    assert df.height == 3
    assert set(df.columns) == {"pair", "date", "rate", "source", "fetched_at"}
    assert df["source"].unique().to_list() == ["frankfurter"]


def test_write_empty_noop(tmp_path: Path) -> None:
    store = FxRateStore(tmp_path)
    written = store.write(pl.DataFrame(schema=FX_PROVIDER_SCHEMA), source="x")
    assert written == 0


def test_read_empty_when_no_data(tmp_path: Path) -> None:
    store = FxRateStore(tmp_path)
    df = store.read(
        "EURUSD",
        "frankfurter",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 5, tzinfo=UTC),
    )
    assert df.height == 0


def test_available_sources(tmp_path: Path) -> None:
    store = FxRateStore(tmp_path)
    store.write(_df(), source="frankfurter")
    assert store.available_sources("EURUSD") == ["frankfurter"]
    assert store.available_sources("USDTUSD") == []
