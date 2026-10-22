"""
__init__.py

Public surface of the evaluation layer.

Heavy modules (`g0`, `walk_forward`) that depend on `backtest` and
`baselines` are exported lazily via `__getattr__` (PEP 562) to avoid
circular imports at package load time. Users can still write
`from locus.evaluation import build_g0_report`; the module is loaded on
first access.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from locus.evaluation.g0 import (
        G0_CRITERION_V1,
        G0Criterion,
        G0Report,
        StrategyG0Report,
        build_g0_report,
    )
    from locus.evaluation.walk_forward import (
        FoldResult,
        WalkForwardBaselineEvaluator,
        WalkForwardStrategyResult,
    )

__all__ = [
    "ExperimentRecord",
    "ExperimentRegistry",
    "FoldResult",
    "G0_CRITERION_V1",
    "G0Criterion",
    "G0Report",
    "PerformanceMetrics",
    "StrategyG0Report",
    "WalkForwardBaselineEvaluator",
    "WalkForwardStrategyResult",
    "block_bootstrap_sharpe_ci",
    "build_g0_report",
    "compute_metrics",
    "deflated_sharpe_ratio",
    "minimum_backtest_length",
    "probability_of_backtest_overfitting",
    "significance_summary",
]


_LAZY_G0: frozenset[str] = frozenset(
    {
        "G0_CRITERION_V1",
        "G0Criterion",
        "G0Report",
        "StrategyG0Report",
        "build_g0_report",
    }
)
_LAZY_WF: frozenset[str] = frozenset(
    {
        "FoldResult",
        "WalkForwardBaselineEvaluator",
        "WalkForwardStrategyResult",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_G0:
        from locus.evaluation import g0

        return getattr(g0, name)
    if name in _LAZY_WF:
        from locus.evaluation import walk_forward

        return getattr(walk_forward, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
