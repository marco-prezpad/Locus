"""
test_converter.py

FxConverter: routing, multi-hop conversions, weekend fallback, staleness,
audit details.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from locus.data.fx.base import FX_PROVIDER_SCHEMA
from locus.data.fx.converter import FxConverter
from locus.data.fx.store import FxRateStore
from locus.data.fx.types import FxRateNotFoundError


def _seed_store(tmp_path: Path) -> FxRateStore:
    """EURUSD ~1.10, USDTUSD ~1.00, three days from 2024-01-01."""
    store = FxRateStore(tmp_path)
    eurusd_rows = [
        {
            "pair": "EURUSD",
            "date": datetime(2024, 1, 1 + i, tzinfo=UTC),
            "rate": 1.10 + 0.01 * i,
        }
        for i in range(3)
    ]
    usdtusd_rows = [
        {
            "pair": "USDTUSD",
            "date": datetime(2024, 1, 1 + i, tzinfo=UTC),
            "rate": 0.999 + 0.001 * i,
        }
        for i in range(3)
    ]
    store.write(pl.DataFrame(eurusd_rows, schema=FX_PROVIDER_SCHEMA), "frankfurter")
    store.write(pl.DataFrame(usdtusd_rows, schema=FX_PROVIDER_SCHEMA), "kraken")
    return store


def _converter(store: FxRateStore) -> FxConverter:
    return FxConverter(
        store=store,
        stale_days_threshold=7,
        source_preferences={
            "EURUSD": ["frankfurter"],
            "USDTUSD": ["kraken"],
        },
    )


def test_same_currency_returns_amount(tmp_path: Path) -> None:
    c = _converter(_seed_store(tmp_path))
    assert c.convert(100.0, "USD", "USD", datetime(2024, 1, 1, tzinfo=UTC)) == 100.0


def test_eur_to_usd_direct(tmp_path: Path) -> None:
    c = _converter(_seed_store(tmp_path))
    result = c.convert_with_details(100.0, "EUR", "USD", datetime(2024, 1, 1, tzinfo=UTC))
    assert result.rate == pytest.approx(1.10)
    assert result.amount_out == pytest.approx(110.0)
    assert len(result.path) == 1


def test_usd_to_eur_inverse(tmp_path: Path) -> None:
    c = _converter(_seed_store(tmp_path))
    result = c.convert_with_details(110.0, "USD", "EUR", datetime(2024, 1, 1, tzinfo=UTC))
    assert result.rate == pytest.approx(1.0 / 1.10)
    assert result.amount_out == pytest.approx(100.0)


def test_eur_to_usdt_two_hops(tmp_path: Path) -> None:
    c = _converter(_seed_store(tmp_path))
    result = c.convert_with_details(100.0, "EUR", "USDT", datetime(2024, 1, 1, tzinfo=UTC))
    # EUR -> USD: *1.10, USD -> USDT: /0.999
    assert result.rate == pytest.approx(1.10 / 0.999)
    assert len(result.path) == 2


def test_usdt_to_eur_two_hops(tmp_path: Path) -> None:
    c = _converter(_seed_store(tmp_path))
    result = c.convert_with_details(100.0, "USDT", "EUR", datetime(2024, 1, 1, tzinfo=UTC))
    # USDT -> USD: *0.999, USD -> EUR: /1.10
    assert result.rate == pytest.approx(0.999 / 1.10)


def test_weekend_falls_back_to_friday(tmp_path: Path) -> None:
    c = _converter(_seed_store(tmp_path))
    # Request Saturday 2024-01-06; last rate is 2024-01-03
    result = c.convert_with_details(100.0, "EUR", "USD", datetime(2024, 1, 6, tzinfo=UTC))
    assert result.date_used == datetime(2024, 1, 3, tzinfo=UTC)
    assert result.rate == pytest.approx(1.12)


def test_stale_rate_raises(tmp_path: Path) -> None:
    c = _converter(_seed_store(tmp_path))
    # Request 30 days later; no rate within 7-day window
    with pytest.raises(FxRateNotFoundError, match="within 7 days"):
        c.convert(100.0, "EUR", "USD", datetime(2024, 2, 1, tzinfo=UTC))


def test_unknown_currency_raises(tmp_path: Path) -> None:
    c = _converter(_seed_store(tmp_path))
    with pytest.raises(FxRateNotFoundError, match="no FX rates"):
        c.convert(100.0, "GBP", "USD", datetime(2024, 1, 1, tzinfo=UTC))


def test_naive_date_raises(tmp_path: Path) -> None:
    c = _converter(_seed_store(tmp_path))
    with pytest.raises(ValueError):
        c.convert(100.0, "EUR", "USD", datetime(2024, 1, 1))


def test_details_include_sources(tmp_path: Path) -> None:
    c = _converter(_seed_store(tmp_path))
    result = c.convert_with_details(100.0, "EUR", "USDT", datetime(2024, 1, 1, tzinfo=UTC))
    assert result.sources["EURUSD"] == "frankfurter"
    assert result.sources["USDTUSD"] == "kraken"
