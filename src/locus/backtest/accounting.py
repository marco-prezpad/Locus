"""
accounting.py

PortfolioAccounting tracks cash and positions. Pure state; the
broker mutates it via `apply_fill`, the engine reads it via `equity`.
`apply_funding` mutates cash with the funding cash flow, independent
of fills and execution costs.

`equity(prices)` raises `MissingPriceError` if a position's asset has no
price. Valuing a missing asset at 0 silently produces spurious −100%
returns; failing loudly is the only defensible behavior.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass, field


class MissingPriceError(Exception):
    """Raised when `equity()` is called for an asset with a non-zero
    position and no price available at the valuation timestamp."""


@dataclass(slots=True)
class PortfolioAccounting:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)

    def get_position(self, asset_id: str) -> float:
        return self.positions.get(asset_id, 0.0)

    def apply_fill(self, asset_id: str, delta_qty: float, fill_price: float) -> None:
        """Apply a signed quantity change. Positive = buy, negative = sell.
        Cash flow is `-delta_qty * fill_price`."""
        self.cash -= delta_qty * fill_price
        new_qty = self.positions.get(asset_id, 0.0) + delta_qty
        if abs(new_qty) < 1e-12:
            self.positions.pop(asset_id, None)
        else:
            self.positions[asset_id] = new_qty

    def apply_funding(
        self,
        asset_id: str,
        qty: float,
        price: float,
        funding_rate: float,
    ) -> float:
        """Apply the funding cash flow for a position.

        Convention: positive funding rate => longs pay shorts.
            cash_flow = -qty * price * funding_rate

        Returns the signed cash flow applied.
        """
        cash_flow = -qty * price * funding_rate
        self.cash += cash_flow
        return cash_flow

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus mark-to-market of positions.

        Raises `MissingPriceError` if any open position has no price.
        """
        value = self.cash
        for asset, qty in self.positions.items():
            if qty == 0:
                continue
            if asset not in prices:
                raise MissingPriceError(
                    f"position {qty} in {asset!r} has no price at "
                    f"valuation timestamp; refusing to value at 0"
                )
            value += qty * prices[asset]
        return value
