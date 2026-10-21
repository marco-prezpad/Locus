"""
__init__.py

Public surface of the features layer. Importing from `locus.features`
registers all built-in feature families on first import (each feature
module calls `register_feature` at module load time).

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from locus.features import (  # noqa: F401 — register
    cross_sectional,
    extended,
    returns,
    trend,
    volatility,
)
from locus.features.base import (
    Feature,
    FeatureRegistry,
    FeatureSpec,
    register_feature,
)
from locus.features.cross_sectional import (
    CrossSectionalPercentileRankFeature,
    CrossSectionalZScoreFeature,
    RelativeMomentumVsMarketFeature,
)
from locus.features.extended import (
    ATRPercentFeature,
    EMADistanceFeature,
    RelativeVolumeFeature,
    VolatilityRatioFeature,
    VolumeZScoreFeature,
)
from locus.features.pipeline import FeaturePipeline
from locus.features.returns import LogReturnFeature, ReturnFeature
from locus.features.trend import MovingAverageDistanceFeature
from locus.features.volatility import RollingVolatilityFeature

__all__ = [
    "ATRPercentFeature",
    "CrossSectionalPercentileRankFeature",
    "CrossSectionalZScoreFeature",
    "EMADistanceFeature",
    "Feature",
    "FeaturePipeline",
    "FeatureRegistry",
    "FeatureSpec",
    "LogReturnFeature",
    "MovingAverageDistanceFeature",
    "RelativeMomentumVsMarketFeature",
    "RelativeVolumeFeature",
    "ReturnFeature",
    "RollingVolatilityFeature",
    "VolatilityRatioFeature",
    "VolumeZScoreFeature",
    "register_feature",
]
