"""
__init__.py

Public surface of the evaluation layer.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from locus.evaluation.metrics import PerformanceMetrics, compute_metrics

__all__ = ["PerformanceMetrics", "compute_metrics"]
