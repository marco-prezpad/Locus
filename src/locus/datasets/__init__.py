"""
__init__.py

Public surface of the datasets layer.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from locus.datasets.dataset import (
    TemporalDataset,
    TrainingDatasetBuilder,
    WalkForwardDataset,
)
from locus.datasets.labels import LabelBuilder
from locus.datasets.splits import (
    PurgedWalkForwardSplitter,
    TemporalSplitter,
)
from locus.datasets.types import (
    DatasetSplit,
    LabelSpec,
    WalkForwardFold,
)
from locus.datasets.uniqueness import (
    compute_uniqueness_weights,
    estimate_n_eff_simple,
)
from locus.datasets.validation import (
    DatasetValidationError,
    validate_temporal_dataset,
)

__all__ = [
    "DatasetSplit",
    "DatasetValidationError",
    "LabelBuilder",
    "LabelSpec",
    "PurgedWalkForwardSplitter",
    "TemporalDataset",
    "TemporalSplitter",
    "TrainingDatasetBuilder",
    "WalkForwardDataset",
    "WalkForwardFold",
    "compute_uniqueness_weights",
    "estimate_n_eff_simple",
    "validate_temporal_dataset",
]
