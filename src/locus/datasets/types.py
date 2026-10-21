"""
types.py

Dataset-layer types: label specs, split results, fold descriptions.
These are the contracts between LabelBuilder, splitters,
TrainingDatasetBuilder, and DatasetValidator.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl


@dataclass(frozen=True, slots=True)
class LabelSpec:
    """One label: `future_return_<name>` over `bars` bars.

    Semantics: close[t + bars] / close[t] - 1, per asset. Uses future
    data (that's what labels do); the leakage guarantee is enforced by
    the splitter, not by the label builder.
    """

    name: str
    bars: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.bars <= 0:
            raise ValueError("bars must be positive")


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """A named temporal split of the dataset with uniqueness weights
    and effective sample size.

    `weights` has one row per sample in `df`, computed as 1/concurrency
    per asset over `horizon_bars`. `n_eff = sum(weights)`, i.e., the
    number of independent observations after accounting for overlap.
    """

    name: str
    df: pl.DataFrame
    weights: pl.Series
    n_eff: float
    start: datetime
    end: datetime

    @property
    def n_samples(self) -> int:
        return self.df.height


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One walk-forward fold: contiguous train/val/test windows with
    purge gaps between them."""

    fold: int
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
