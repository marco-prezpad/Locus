"""
config.py

BacktestConfig combines everything the engine needs: capital, cost
model, execution constraints (min notional, lot sizes), time parameters,
and whether funding cash flows are applied when a funding DataFrame is
provided.

Loaders:
- `load_capital_scenario(name)`: reads `configs/capital/*.yaml`.
- `load_backtest_config(capital_name, strategy_name, cost_scenario)`:
  composes a full BacktestConfig.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from locus.backtest.costs import TransactionCostModel, load_cost_model

_DEFAULT_LOT_SIZES: dict[str, float] = {
    "BTC": 0.00001,
    "ETH": 0.0001,
    "SOL": 0.001,
    "BNB": 0.001,
    "XRP": 0.1,
    "ADA": 1.0,
    "DOGE": 1.0,
    "AVAX": 0.001,
    "DOT": 0.01,
    "LINK": 0.01,
}


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Everything the engine needs to run a backtest."""

    strategy_name: str
    capital: float
    currency: str
    cost_model: TransactionCostModel
    min_order_notional: float
    default_lot_size: float
    lot_sizes: dict[str, float]
    bars_per_day: int = 24
    rebalance_freq_bars: int = 24
    risk_free_rate_annual: float = 0.0
    apply_funding: bool = True  # if funding is provided, apply it

    def __post_init__(self) -> None:
        if self.capital <= 0:
            raise ValueError("capital must be positive")
        if self.min_order_notional < 0:
            raise ValueError("min_order_notional must be non-negative")
        if self.bars_per_day <= 0:
            raise ValueError("bars_per_day must be positive")
        if self.rebalance_freq_bars <= 0:
            raise ValueError("rebalance_freq_bars must be positive")

    @property
    def periods_per_year(self) -> int:
        return int((self.bars_per_day * 365) / self.rebalance_freq_bars)

    def lot_size(self, asset_id: str) -> float:
        return self.lot_sizes.get(asset_id, self.default_lot_size)


def load_capital_scenario(name: str) -> dict:
    """Read `configs/capital/*.yaml`."""
    path = Path(f"configs/capital/{name}.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Capital scenario not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_backtest_config(
    strategy_name: str,
    capital_name: Literal["1", "10", "100", "1k", "10k", "100k", "1m"],
    cost_scenario: Literal["optimistic", "central", "pessimistic"] = "central",
    bars_per_day: int = 24,
    rebalance_freq_bars: int = 24,
    risk_free_rate_annual: float = 0.0,
) -> BacktestConfig:
    cap = load_capital_scenario(capital_name)
    return BacktestConfig(
        strategy_name=strategy_name,
        capital=float(cap["capital"]),
        currency=cap.get("currency", "USDT"),
        cost_model=load_cost_model(cost_scenario),
        min_order_notional=float(cap["min_order_notional"]),
        default_lot_size=1e-8,
        lot_sizes=dict(_DEFAULT_LOT_SIZES),
        bars_per_day=bars_per_day,
        rebalance_freq_bars=rebalance_freq_bars,
        risk_free_rate_annual=risk_free_rate_annual,
    )
