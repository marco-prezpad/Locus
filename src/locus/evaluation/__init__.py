"""
__init__.py

Public surface of the evaluation layer.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from locus.evaluation.experiments import (
    ExperimentRecord,
    ExperimentRegistry,
)
from locus.evaluation.metrics import PerformanceMetrics, compute_metrics
from locus.evaluation.significance import (
    block_bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    minimum_backtest_length,
    probability_of_backtest_overfitting,
    significance_summary,
)

__all__ = [
    "ExperimentRecord",
    "ExperimentRegistry",
    "PerformanceMetrics",
    "block_bootstrap_sharpe_ci",
    "compute_metrics",
    "deflated_sharpe_ratio",
    "minimum_backtest_length",
    "probability_of_backtest_overfitting",
    "significance_summary",
]
