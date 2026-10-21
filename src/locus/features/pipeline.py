"""
pipeline.py

FeaturePipeline composes a list of Feature instances and applies them
sequentially. Each feature returns the DataFrame with one extra column;
the pipeline chains them.

Row policy: the pipeline does NOT drop null rows. Features produce nulls
at the start of each asset's history (insufficient lookback). F4 drops
rows with null features when building the training dataset; keeping
nulls here makes the pipeline idempotent and leak-testable.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

import polars as pl

from locus.features.base import Feature, FeatureRegistry, FeatureSpec
from locus.infrastructure.logging import LoggerFactory


class FeaturePipeline:
    """Apply a list of Feature instances in order."""

    def __init__(self, features: list[Feature]) -> None:
        if not features:
            raise ValueError("features must be non-empty")
        names = [f.name for f in features]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate feature names: {sorted(duplicates)}")
        self._features = features
        self._log = LoggerFactory.get("features.pipeline")

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self._features]

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all features to df. The output has the BAR_SCHEMA
        columns plus one column per feature, sorted by
        (asset_id, timestamp)."""
        if df.height == 0:
            return df

        result = df.sort(["asset_id", "timestamp"])
        for feature in self._features:
            result = feature.compute(result)

        self._log.info(
            "features_computed",
            extra={
                "rows": result.height,
                "assets": result["asset_id"].n_unique(),
                "features": len(self._features),
            },
        )
        return result

    @classmethod
    def from_registry(cls, registry: FeatureRegistry) -> FeaturePipeline:
        return cls(registry.build())

    @classmethod
    def from_specs(cls, specs: list[FeatureSpec]) -> FeaturePipeline:
        return cls.from_registry(FeatureRegistry(specs))
