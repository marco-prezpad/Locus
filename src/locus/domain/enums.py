"""
enums.py

Enumerations describing market types, execution primitives, risk
violations, and the three cost scenarios. Values are strings so they
serialize cleanly to JSON and persist stably in storage.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from enum import StrEnum


class MarketType(StrEnum):
    """Asset class the system can trade."""

    CRYPTO = "crypto"
    EQUITIES = "equities"
    ETF = "etf"
    FOREX = "forex"
    FUTURES = "futures"


class OrderSide(StrEnum):
    """Direction of an order or fill."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Order type understood by the execution layer."""

    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(StrEnum):
    """State-machine states for an order."""

    CREATED = "created"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ViolationType(StrEnum):
    """Categories of risk-limit violations."""

    POSITION_SIZE = "position_size"
    PORTFOLIO_VOLATILITY = "portfolio_volatility"
    DRAWDOWN = "drawdown"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"
    KILL_SWITCH = "kill_switch"


class CostScenarioName(StrEnum):
    """The three pre-registered cost scenarios."""

    OPTIMISTIC = "optimistic"
    CENTRAL = "central"
    PESSIMISTIC = "pessimistic"


class ImpactModel(StrEnum):
    """Market-impact model used by the backtester."""

    SQRT_VOLUME = "sqrt_volume"
    LINEAR = "linear"
    NONE = "none"
