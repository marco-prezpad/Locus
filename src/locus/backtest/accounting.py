"""
accounting.py

PortfolioAccounting tracks cash and positions. Pure state; the broker
mutates it via `apply_fill`, the engine reads it via `equity`.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PortfolioAccounting:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)

    def get_position(self, asset_id: str) -> float:
        return self.positions.get(asset_id, 0.0)

    def apply_fill(self, asset_id: str, delta_qty: float, fill_price: float) -> None:
        """Apply a signed quantity change. Positive = buy, negative =
        sell. Cash flow is `-delta_qty * fill_price` (pay on buy,
        receive on sell). The fill price already includes the cost
        adjustment (buy above open, sell below open)."""
        self.cash -= delta_qty * fill_price
        new_qty = self.positions.get(asset_id, 0.0) + delta_qty
        if abs(new_qty) < 1e-12:
            self.positions.pop(asset_id, None)
        else:
            self.positions[asset_id] = new_qty

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus mark-to-market of positions. Assets without a price
        are valued at zero (they contribute nothing)."""
        value = self.cash
        for asset, qty in self.positions.items():
            if asset in prices:
                value += qty * prices[asset]
        return value
