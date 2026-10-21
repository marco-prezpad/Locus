"""
costs.py

Cost models for the event-driven backtester: FeeModel, SpreadModel,
SlippageModel, MarketImpactModel, and the composite TransactionCostModel.

All models return cost in basis points (bps) per trade. The composite
`total_cost_bps` is the sum of the four components.

Conventions:
- **Fee**: per-trade fee in bps (maker/taker). Flat by default; can be
  parameterised per venue when multi-exchange lands (B4).
- **Spread**: half-spread per trade, referencing mid. A `spread_bps`
  of 5 means you pay 2.5 bps when crossing the spread, either side.
- **Slippage**: extra bps per trade for adverse selection, latency,
  and imperfect fills. Conservative default 5 bps central.
- **Market impact**: participation-based. Formula
  `impact_bps = k * sigma_daily * sqrt(Q / V_daily) * 1e4`, where Q is
  the order notional and V_daily is the daily traded notional (both
  in quote currency). `k` is the calibration parameter; it must NOT
  be tuned to improve backtests (see plan §8.13).

The `load_cost_model` helper reads one of the three pre-registered
scenarios in `configs/costs/{optimistic,central,pessimistic}.yaml`.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from locus.infrastructure.config import load_cost_scenario


@dataclass(frozen=True, slots=True)
class FeeModel:
    """Per-trade fee in basis points."""

    fee_bps: float

    def __post_init__(self) -> None:
        if self.fee_bps < 0:
            raise ValueError("fee_bps must be non-negative")

    def cost_bps(self, notional: float) -> float:
        return self.fee_bps


@dataclass(frozen=True, slots=True)
class SpreadModel:
    """Half-spread per trade, referencing mid.

    A `spread_bps` of X means the quoted spread is X bps and you pay
    X/2 bps per trade (because you cross from mid to the bid/ask).
    Round-trip cost is X bps.
    """

    spread_bps: float

    def __post_init__(self) -> None:
        if self.spread_bps < 0:
            raise ValueError("spread_bps must be non-negative")

    def cost_bps(self, notional: float) -> float:
        return self.spread_bps / 2.0


@dataclass(frozen=True, slots=True)
class SlippageModel:
    """Additional slippage per trade in basis points.

    Covers adverse selection, latency, and imperfect fills beyond the
    spread. Conservative default 5 bps central; 15 bps pessimistic.
    """

    slippage_bps: float

    def __post_init__(self) -> None:
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")

    def cost_bps(self, notional: float) -> float:
        return self.slippage_bps


@dataclass(frozen=True, slots=True)
class MarketImpactModel:
    """Participation-based market impact in basis points.

    Formula (sqrt_volume, the MVP default):
        impact_bps = k * sigma_daily * sqrt(Q / V_daily) * 1e4

    where:
    - Q: order notional (quote currency, e.g. USDT).
    - V_daily: daily traded notional (quote currency).
    - sigma_daily: daily volatility as a fraction (e.g. 0.03).
    - k: calibration parameter (~1.0 central, 0.5 optimistic, 1.5
      pessimistic).

    `k` must NOT be tuned to improve backtests. Calibrate against data
    or take from conservative literature (plan §8.13). The three
    scenarios provide a range for sensitivity analysis.

    `linear` and `none` are available for contrast but are not the MVP
    default.
    """

    impact_k: float
    model: Literal["sqrt_volume", "linear", "none"] = "sqrt_volume"

    def __post_init__(self) -> None:
        if self.impact_k < 0:
            raise ValueError("impact_k must be non-negative")
        if self.model not in ("sqrt_volume", "linear", "none"):
            raise ValueError(f"unknown impact model: {self.model}")

    def cost_bps(
        self,
        notional: float,
        daily_volume_quote: float,
        sigma_daily: float,
    ) -> float:
        if self.model == "none" or notional <= 0 or daily_volume_quote <= 0:
            return 0.0
        participation = notional / daily_volume_quote
        if self.model == "linear":
            return float(self.impact_k * sigma_daily * participation * 1e4)
        # sqrt_volume
        return float(self.impact_k * sigma_daily * np.sqrt(participation) * 1e4)


@dataclass(frozen=True, slots=True)
class TransactionCostModel:
    """Composite model: per-trade cost = fee + half-spread + slippage
    + market impact, all in bps."""

    fee: FeeModel
    spread: SpreadModel
    slippage: SlippageModel
    impact: MarketImpactModel

    def total_cost_bps(
        self,
        notional: float,
        daily_volume_quote: float,
        sigma_daily: float,
    ) -> float:
        return (
            self.fee.cost_bps(notional)
            + self.spread.cost_bps(notional)
            + self.slippage.cost_bps(notional)
            + self.impact.cost_bps(notional, daily_volume_quote, sigma_daily)
        )

    def cost_breakdown(
        self,
        notional: float,
        daily_volume_quote: float,
        sigma_daily: float,
    ) -> dict[str, float]:
        """Return the contribution of each component in bps. Useful for
        diagnostics and attribution when comparing scenarios."""
        return {
            "fee": self.fee.cost_bps(notional),
            "spread": self.spread.cost_bps(notional),
            "slippage": self.slippage.cost_bps(notional),
            "impact": self.impact.cost_bps(notional, daily_volume_quote, sigma_daily),
        }


def load_cost_model(
    scenario: Literal["optimistic", "central", "pessimistic"],
) -> TransactionCostModel:
    """Load one of the three pre-registered cost scenarios.

    Reads from `configs/costs/<scenario>.yaml`. Raises
    FileNotFoundError if the scenario file does not exist.
    """
    data = load_cost_scenario(scenario)
    return TransactionCostModel(
        fee=FeeModel(fee_bps=float(data["fee_bps"])),
        spread=SpreadModel(spread_bps=float(data["spread_bps"])),
        slippage=SlippageModel(slippage_bps=float(data["slippage_bps"])),
        impact=MarketImpactModel(impact_k=float(data["impact_k"])),
    )
