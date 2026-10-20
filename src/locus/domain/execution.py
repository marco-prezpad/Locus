"""
execution.py

Execution primitives: orders, fills, positions, and portfolio state.
`PortfolioState.cash` is a single float in the settlement currency
(`CapitalScenario.currency`). Multi-currency holdings are out of scope
for the MVP and will be added when the FX layer becomes currency-aware
in B5.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass, field

from locus.domain.enums import OrderSide, OrderStatus, OrderType
from locus.domain.types import AssetId, Timestamp


@dataclass(frozen=True, slots=True)
class Order:
    """An instruction to buy or sell a quantity of an asset."""

    order_id: str
    asset_id: AssetId
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: float | None
    created_at: Timestamp
    status: OrderStatus = OrderStatus.CREATED
    filled_quantity: float = 0.0

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("order_id must be non-empty")
        if self.quantity <= 0:
            raise ValueError("quantity must be strictly positive")
        if self.filled_quantity < 0:
            raise ValueError("filled_quantity must be non-negative")
        if self.filled_quantity > self.quantity:
            raise ValueError("filled_quantity must not exceed quantity")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders require a limit_price")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be strictly positive")


@dataclass(frozen=True, slots=True)
class Fill:
    """An execution report: what was filled, at what price, and when."""

    order_id: str
    asset_id: AssetId
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: Timestamp

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("order_id must be non-empty")
        if self.quantity <= 0:
            raise ValueError("fill quantity must be strictly positive")
        if self.price <= 0:
            raise ValueError("fill price must be strictly positive")
        if self.commission < 0:
            raise ValueError("commission must be non-negative")


@dataclass(frozen=True, slots=True)
class Position:
    """An open position in one asset: signed quantity and average cost."""

    asset_id: AssetId
    quantity: float
    average_cost: float

    def __post_init__(self) -> None:
        if self.average_cost < 0:
            raise ValueError("average_cost must be non-negative")
        if self.quantity != 0 and self.average_cost == 0:
            raise ValueError("non-zero position requires a non-zero average_cost")


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Snapshot of a portfolio: cash (single settlement currency) and positions."""

    timestamp: Timestamp
    cash: float
    positions: dict[AssetId, Position] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cash < 0:
            raise ValueError("cash must be non-negative")

    @property
    def total_positions(self) -> int:
        """Number of non-zero positions currently held."""
        return sum(1 for p in self.positions.values() if p.quantity != 0)
