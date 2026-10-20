"""
__init__.py

Public surface of the domain layer. Importing from `locus.domain` gives
access to every contract without reaching into individual modules.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from locus.domain.asset import Asset
from locus.domain.enums import (
    CostScenarioName,
    ImpactModel,
    MarketType,
    OrderSide,
    OrderStatus,
    OrderType,
    ViolationType,
)
from locus.domain.execution import Fill, Order, PortfolioState, Position
from locus.domain.market_data import OHLCVBar, OrderBookSnapshot, Trade
from locus.domain.portfolio import CapitalScenario, TargetPortfolio
from locus.domain.predictions import OpportunityScore, Prediction
from locus.domain.risk import RiskDecision, RiskViolation
from locus.domain.signal_execution import SignalExecutionContract
from locus.domain.types import AssetId, Currency, Timestamp

__all__ = [
    "Asset",
    "AssetId",
    "CapitalScenario",
    "CostScenarioName",
    "Currency",
    "Fill",
    "ImpactModel",
    "MarketType",
    "OHLCVBar",
    "OpportunityScore",
    "Order",
    "OrderBookSnapshot",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PortfolioState",
    "Position",
    "Prediction",
    "RiskDecision",
    "RiskViolation",
    "SignalExecutionContract",
    "TargetPortfolio",
    "Timestamp",
    "Trade",
    "ViolationType",
]
