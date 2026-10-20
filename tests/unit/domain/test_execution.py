"""
test_execution.py

Verifies order, fill, position, and portfolio-state invariants.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from locus.domain import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioState,
    Position,
)
from locus.domain.types import AssetId, Timestamp


def _ts() -> Timestamp:
    return Timestamp(datetime(2026, 1, 1, tzinfo=UTC))


def _order(**overrides: object) -> Order:
    defaults = {
        "order_id": "ord-1",
        "asset_id": AssetId("BTC"),
        "side": OrderSide.BUY,
        "order_type": OrderType.MARKET,
        "quantity": 1.0,
        "limit_price": None,
        "created_at": _ts(),
    }
    defaults.update(overrides)
    return Order(**defaults)  # type: ignore[arg-type]


def test_order_market_valid() -> None:
    o = _order()
    assert o.status == OrderStatus.CREATED


def test_order_limit_requires_price() -> None:
    with pytest.raises(ValueError):
        _order(order_type=OrderType.LIMIT, limit_price=None)


def test_order_rejects_overfilled() -> None:
    with pytest.raises(ValueError):
        _order(quantity=1.0, filled_quantity=2.0)


def test_order_rejects_non_positive_quantity() -> None:
    with pytest.raises(ValueError):
        _order(quantity=0.0)


def test_fill_valid() -> None:
    f = Fill(
        order_id="ord-1",
        asset_id=AssetId("BTC"),
        side=OrderSide.BUY,
        quantity=1.0,
        price=100.0,
        commission=0.1,
        timestamp=_ts(),
    )
    assert f.commission == pytest.approx(0.1)


def test_fill_rejects_negative_commission() -> None:
    with pytest.raises(ValueError):
        Fill(
            order_id="ord-1",
            asset_id=AssetId("BTC"),
            side=OrderSide.BUY,
            quantity=1.0,
            price=100.0,
            commission=-0.1,
            timestamp=_ts(),
        )


def test_position_valid() -> None:
    p = Position(asset_id=AssetId("BTC"), quantity=1.0, average_cost=100.0)
    assert p.quantity == 1.0


def test_position_requires_average_cost_if_open() -> None:
    with pytest.raises(ValueError):
        Position(asset_id=AssetId("BTC"), quantity=1.0, average_cost=0.0)


def test_portfolio_state_valid() -> None:
    ps = PortfolioState(
        timestamp=_ts(),
        cash=1000.0,
        positions={
            AssetId("BTC"): Position(AssetId("BTC"), 1.0, 100.0),
            AssetId("ETH"): Position(AssetId("ETH"), 0.0, 0.0),
        },
    )
    assert ps.total_positions == 1


def test_portfolio_state_rejects_negative_cash() -> None:
    with pytest.raises(ValueError):
        PortfolioState(timestamp=_ts(), cash=-1.0)
