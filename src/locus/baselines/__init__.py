"""
__init__.py

Public surface of the baselines layer.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from locus.baselines.base import BaselineStrategy, EvaluationResult
from locus.baselines.comparison import comparison_table, run_baselines
from locus.baselines.evaluator import VectorizedEvaluator
from locus.baselines.strategies import (
    BuyAndHoldBTC,
    EqualWeightUniverse,
    MomentumTopK,
    RandomTopK,
)

__all__ = [
    "BaselineStrategy",
    "BuyAndHoldBTC",
    "EqualWeightUniverse",
    "EvaluationResult",
    "MomentumTopK",
    "RandomTopK",
    "VectorizedEvaluator",
    "comparison_table",
    "run_baselines",
]
