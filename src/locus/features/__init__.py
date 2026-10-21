"""
__init__.py

Public surface of the features layer. Importing from `locus.features`
registers all built-in feature families on first import (each feature
module calls `register_feature` at module load time).

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from locus.features import returns, trend, volatility  # noqa: F401 — register
from locus.features.base import (
    Feature,
    FeatureRegistry,
    FeatureSpec,
    register_feature,
)
from locus.features.pipeline import FeaturePipeline
from locus.features.returns import LogReturnFeature, ReturnFeature
from locus.features.trend import MovingAverageDistanceFeature
from locus.features.volatility import RollingVolatilityFeature

__all__ = [
    "Feature",
    "FeaturePipeline",
    "FeatureRegistry",
    "FeatureSpec",
    "LogReturnFeature",
    "MovingAverageDistanceFeature",
    "ReturnFeature",
    "RollingVolatilityFeature",
    "register_feature",
]
