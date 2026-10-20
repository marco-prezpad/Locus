"""
test_market_data.py

Verifies the OHLCV, Trade, and OrderBookSnapshot invariants are enforced
at construction time.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from locus.domain import OHLCVBar, OrderBookSnapshot, OrderSide, Trade
from locus.domain.types import AssetId, Timestamp


def _ts(minutes: int = 0) -> Timestamp:
    return Timestamp(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes))


def _bar(**overrides: float | Timestamp) -> OHLCVBar:
    defaults = {
        "asset_id": AssetId("BTC"),
        "start": _ts(0),
        "end": _ts(60),
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 105.0,
        "volume": 1000.0,
    }
    defaults.update(overrides)  # type: ignore[arg-type]
    return OHLCVBar(**defaults)  # type: ignore[arg-type]


def test_bar_valid() -> None:
    bar = _bar()
    assert bar.high >= bar.close >= bar.low


def test_bar_rejects_end_before_start() -> None:
    with pytest.raises(ValueError):
        _bar(start=_ts(60), end=_ts(0))


def test_bar_rejects_non_positive_price() -> None:
    with pytest.raises(ValueError):
        _bar(open=0.0)


def test_bar_rejects_high_below_close() -> None:
    with pytest.raises(ValueError):
        _bar(high=100.0, close=105.0)


def test_bar_rejects_low_above_open() -> None:
    with pytest.raises(ValueError):
        _bar(low=102.0, open=100.0)


def test_bar_rejects_negative_volume() -> None:
    with pytest.raises(ValueError):
        _bar(volume=-1.0)


def test_trade_valid() -> None:
    t = Trade(
        asset_id=AssetId("BTC"),
        timestamp=_ts(),
        price=100.0,
        quantity=1.5,
        side=OrderSide.BUY,
    )
    assert t.side == OrderSide.BUY


def test_trade_rejects_non_positive_price() -> None:
    with pytest.raises(ValueError):
        Trade(
            asset_id=AssetId("BTC"),
            timestamp=_ts(),
            price=0.0,
            quantity=1.0,
            side=OrderSide.BUY,
        )


def test_order_book_valid() -> None:
    ob = OrderBookSnapshot(
        asset_id=AssetId("BTC"),
        timestamp=_ts(),
        bid_price=99.0,
        bid_size=1.0,
        ask_price=101.0,
        ask_size=1.0,
    )
    assert ob.spread == pytest.approx(2.0)


def test_order_book_rejects_crossed_book() -> None:
    with pytest.raises(ValueError):
        OrderBookSnapshot(
            asset_id=AssetId("BTC"),
            timestamp=_ts(),
            bid_price=101.0,
            bid_size=1.0,
            ask_price=99.0,
            ask_size=1.0,
        )
