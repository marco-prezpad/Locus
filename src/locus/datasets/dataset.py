"""
dataset.py

TrainingDatasetBuilder orchestrates the whole dataset assembly:

    OHLCV → features (F3) → labels (F4) → drop nulls → split (F4)

If labels are provided in `df` (already computed), the builder skips
the label step. This lets the smoke script and tests exercise the
pipeline without recomputing labels.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from locus.datasets.labels import LabelBuilder
from locus.datasets.splits import (
    PurgedWalkForwardSplitter,
    TemporalSplitter,
)
from locus.datasets.types import DatasetSplit, WalkForwardFold
from locus.features.pipeline import FeaturePipeline
from locus.infrastructure.logging import LoggerFactory


@dataclass(frozen=True, slots=True)
class TemporalDataset:
    """Result of a single temporal split (train/val/test)."""

    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    feature_names: list[str]
    label_names: list[str]

    @property
    def all_splits(self) -> list[DatasetSplit]:
        return [self.train, self.val, self.test]


@dataclass(frozen=True, slots=True)
class WalkForwardDataset:
    """Result of a walk-forward split (K folds)."""

    folds: list[WalkForwardFold]
    feature_names: list[str]
    label_names: list[str]


class TrainingDatasetBuilder:
    """Build the dataset: features + labels + splits, dropping nulls."""

    def __init__(
        self,
        feature_pipeline: FeaturePipeline,
        label_builder: LabelBuilder,
        splitter: TemporalSplitter | PurgedWalkForwardSplitter,
    ) -> None:
        self._features = feature_pipeline
        self._labels = label_builder
        self._splitter = splitter
        self._log = LoggerFactory.get("datasets.builder")

    def build_from_ohlcv(self, ohlcv: pl.DataFrame) -> TemporalDataset | WalkForwardDataset:
        """Run the full pipeline: features → labels → drop nulls → splits."""
        self._log.info(
            "building_dataset_started",
            extra={
                "ohlcv_rows": ohlcv.height,
                "feature_count": len(self._features.feature_names),
                "label_count": len(self._labels.label_names),
            },
        )

        with_features = self._features.compute(ohlcv)
        with_labels = self._labels.compute(with_features)

        required = self._features.feature_names + self._labels.label_names
        cleaned = with_labels.drop_nulls(subset=required)

        self._log.info(
            "dataset_cleaned",
            extra={
                "rows_before": with_labels.height,
                "rows_after": cleaned.height,
                "dropped": with_labels.height - cleaned.height,
            },
        )

        return self._apply_split(cleaned)

    def build_from_labelled(self, df: pl.DataFrame) -> TemporalDataset | WalkForwardDataset:
        """Split an already-featurized-and-labelled DataFrame. Skips
        feature/label computation; used by tests and smoke scripts."""
        required = self._features.feature_names + self._labels.label_names
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        cleaned = df.drop_nulls(subset=required)
        return self._apply_split(cleaned)

    def _apply_split(self, df: pl.DataFrame) -> TemporalDataset | WalkForwardDataset:
        if isinstance(self._splitter, TemporalSplitter):
            splits = self._splitter.split(df)
            return TemporalDataset(
                train=splits["train"],
                val=splits["val"],
                test=splits["test"],
                feature_names=self._features.feature_names,
                label_names=self._labels.label_names,
            )
        if isinstance(self._splitter, PurgedWalkForwardSplitter):
            folds = self._splitter.split(df)
            return WalkForwardDataset(
                folds=folds,
                feature_names=self._features.feature_names,
                label_names=self._labels.label_names,
            )
        raise TypeError(f"unknown splitter: {type(self._splitter).__name__}")
