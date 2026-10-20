"""
portfolio.py

Target portfolios and the capital scenario they were built against. A
TargetPortfolio is the portfolio engine's output: weights per asset plus
the cash weight that completes the allocation to 1.

The CapitalScenario carries a `currency` field that identifies the
settlement currency for `capital` and for all downstream PnL reporting.
The default is "USDT", matching the MVP's settlement currency (see
docs/currency.md). The field is extensible to USD, EUR, BTC, etc.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass, field

from locus.domain.enums import CostScenarioName, ImpactModel
from locus.domain.types import AssetId, Timestamp


@dataclass(frozen=True, slots=True)
class CapitalScenario:
    """Capital- and cost-related parameters for one execution scenario.

    `currency` identifies the settlement currency in which `capital`,
    `min_order_notional`, and all downstream PnL are expressed. Defaults
    to "USDT" (the MVP's settlement currency). Valid currencies are
    uppercase ISO-style codes 2-5 characters long (USDT, USD, EUR, BTC,
    ETH, ...).

    `impact_k` is a calibrated coefficient, not a tuning knob: it must
    come from data or from conservative literature, and must be reported
    with its uncertainty.

    The three canonical scenarios are `optimistic`, `central`, and
    `pessimistic`.
    """

    name: CostScenarioName
    currency: str
    capital: float
    min_order_notional: float
    fee_bps: float
    spread_bps: float
    slippage_bps: float
    impact_model: ImpactModel
    impact_k: float
    fee_schedule_version: str

    def __post_init__(self) -> None:
        if not self.currency:
            raise ValueError("currency must be non-empty")
        if not (2 <= len(self.currency) <= 5):
            raise ValueError("currency must be 2-5 characters long")
        if not self.currency.isupper():
            raise ValueError("currency must be uppercase (e.g. USDT, USD, EUR)")
        if not self.currency.isalpha():
            raise ValueError("currency must contain only letters")
        if self.capital <= 0:
            raise ValueError("capital must be strictly positive")
        if self.min_order_notional < 0:
            raise ValueError("min_order_notional must be non-negative")
        for field_name in ("fee_bps", "spread_bps", "slippage_bps", "impact_k"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if not self.fee_schedule_version:
            raise ValueError("fee_schedule_version must be non-empty")


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    """Desired allocation: weights per asset plus a cash weight.

    `weights` maps each `AssetId` to a target fraction of total capital.
    The sum of `weights.values()` plus `cash_weight` must not exceed 1
    (within a small numerical tolerance). `constraints_applied` records
    which risk limits were bound at construction.

    The `capital_scenario` carries the settlement currency in which the
    resulting portfolio is denominated.
    """

    timestamp: Timestamp
    weights: dict[AssetId, float]
    cash_weight: float
    capital_scenario: CapitalScenario
    constraints_applied: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cash_weight < 0:
            raise ValueError("cash_weight must be non-negative")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("all weights must be non-negative")
        total = sum(self.weights.values()) + self.cash_weight
        if total > 1.0 + 1e-9:
            raise ValueError(f"weights plus cash must not exceed 1 (got {total:.9f})")
