"""
broker.py

SimulatedBroker: executes rebalance deltas at the next bar's open,
enforcing min notional and lot sizes, and applying costs.

Execution semantics:
1. For each asset, target notional = current_equity × weight.
2. Target quantity = target_notional / open_price, rounded DOWN to
   lot size.
3. Delta quantity = target - current. If |delta_notional| <
   min_order_notional → skip.
4. Fill price: buy at open × (1 + cost_bps/1e4), sell at
   open × (1 - cost_bps/1e4). Cost is embedded in the price.

Counters `n_skipped_min` and `n_skipped_lot` accumulate across
rebalances; the engine reads them at the end.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

import math
from datetime import datetime

from locus.backtest.accounting import PortfolioAccounting
from locus.backtest.config import BacktestConfig
from locus.backtest.events import FillEvent
from locus.infrastructure.logging import LoggerFactory


class SimulatedBroker:
    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self._cost_model = config.cost_model
        self._min_notional = config.min_order_notional
        self._log = LoggerFactory.get("backtest.broker")
        self.n_skipped_min: int = 0
        self.n_skipped_lot: int = 0

    def execute_rebalance(
        self,
        timestamp: datetime,
        current_equity: float,
        target_weights: dict[str, float],
        open_prices: dict[str, float],
        sigma_daily: dict[str, float],
        daily_volume_quote: dict[str, float],
        accounting: PortfolioAccounting,
    ) -> list[FillEvent]:
        fills: list[FillEvent] = []
        all_assets = set(target_weights.keys()) | set(accounting.positions.keys())

        for asset in all_assets:
            weight = target_weights.get(asset, 0.0)
            price = open_prices.get(asset)
            if price is None or price <= 0:
                continue

            target_notional = current_equity * weight
            target_qty_raw = target_notional / price

            lot = self._config.lot_size(asset)
            target_qty = math.floor(target_qty_raw / lot + 1e-9) * lot

            current_qty = accounting.get_position(asset)
            delta_qty = target_qty - current_qty

            if abs(delta_qty) < lot * 0.5:
                continue  # no meaningful change

            order_notional = abs(delta_qty) * price
            if order_notional < self._min_notional:
                self.n_skipped_min += 1
                self._log.debug(
                    "order_skipped_below_min",
                    extra={
                        "asset": asset,
                        "notional": round(order_notional, 4),
                        "min": self._min_notional,
                    },
                )
                continue

            sigma = sigma_daily.get(asset, 0.0) or 0.0
            vol = daily_volume_quote.get(asset, 0.0) or 0.0
            cost_bps = self._cost_model.total_cost_bps(order_notional, vol, sigma)

            side = "buy" if delta_qty > 0 else "sell"
            adjustment = 1.0 + cost_bps / 1e4 if side == "buy" else 1.0 - cost_bps / 1e4
            fill_price = price * adjustment
            cost_quote = abs(delta_qty) * abs(fill_price - price)

            accounting.apply_fill(asset, delta_qty, fill_price)
            fills.append(
                FillEvent(
                    timestamp=timestamp,
                    asset_id=asset,
                    side=side,
                    quantity=abs(delta_qty),
                    fill_price=fill_price,
                    cost_bps=cost_bps,
                    cost_quote=cost_quote,
                )
            )
        return fills
