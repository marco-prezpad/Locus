"""
__init__.py

Public surface of the backtest layer.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from locus.backtest.accounting import PortfolioAccounting
from locus.backtest.analyzer import report_to_row, reports_table
from locus.backtest.broker import SimulatedBroker
from locus.backtest.config import (
    BacktestConfig,
    load_backtest_config,
    load_capital_scenario,
)
from locus.backtest.costs import (
    FeeModel,
    MarketImpactModel,
    SlippageModel,
    SpreadModel,
    TransactionCostModel,
    load_cost_model,
)
from locus.backtest.engine import BacktestEngine
from locus.backtest.events import BacktestReport, FillEvent
from locus.backtest.market_state import (
    enrich_with_market_state,
    estimate_daily_volume_quote,
    estimate_sigma_daily,
)
from locus.backtest.scaling import CapitalScalingExperiment

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestReport",
    "CapitalScalingExperiment",
    "FeeModel",
    "FillEvent",
    "MarketImpactModel",
    "PortfolioAccounting",
    "SimulatedBroker",
    "SlippageModel",
    "SpreadModel",
    "TransactionCostModel",
    "enrich_with_market_state",
    "estimate_daily_volume_quote",
    "estimate_sigma_daily",
    "load_backtest_config",
    "load_capital_scenario",
    "load_cost_model",
    "report_to_row",
    "reports_table",
]
