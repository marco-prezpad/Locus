"""
validation.py

DatasetValidator: post-hoc checks on the assembled dataset. Fails
loudly if any invariant is violated. Intended to run after
TrainingDatasetBuilder in the smoke script and in tests, not in the
pipeline itself (fast, side-effect-free).

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from locus.datasets.types import DatasetSplit


class DatasetValidationError(Exception):
    """Raised when a dataset invariant is violated."""


def check_no_overlap(
    train: DatasetSplit,
    val: DatasetSplit,
    test: DatasetSplit,
) -> None:
    """Splits must not overlap in time."""
    if train.end >= val.start:
        raise DatasetValidationError(f"train end ({train.end}) >= val start ({val.start})")
    if val.end >= test.start:
        raise DatasetValidationError(f"val end ({val.end}) >= test start ({test.start})")


def check_purge_gap(
    train: DatasetSplit,
    val: DatasetSplit,
    bar_delta: timedelta,
    purge_bars: int,
) -> None:
    """Gap between train.end and val.start must be at least `purge_bars`."""
    gap = val.start - train.end
    if gap < bar_delta * purge_bars:
        raise DatasetValidationError(
            f"purge gap between train and val is {gap}, "
            f"expected >= {bar_delta * purge_bars} ({purge_bars} bars)"
        )


def check_no_nulls(split: DatasetSplit, columns: list[str]) -> None:
    """No nulls in the required columns."""
    null_counts = split.df.select([pl.col(c).null_count().alias(c) for c in columns]).row(0)
    if any(n > 0 for n in null_counts):
        raise DatasetValidationError(
            f"split {split.name!r} has nulls: {dict(zip(columns, null_counts, strict=False))}"
        )


def check_weights_sum(split: DatasetSplit) -> None:
    """Sum of weights must equal `n_eff` (by construction)."""
    total = float(split.weights.sum())
    if abs(total - split.n_eff) > 1e-6:
        raise DatasetValidationError(
            f"split {split.name!r} n_eff mismatch: weights sum = {total}, n_eff = {split.n_eff}"
        )


def check_label_is_forward_return(
    df: pl.DataFrame,
    label_name: str,
    bars: int,
    tolerance: float = 1e-9,
) -> None:
    """Sanity check: label at t equals close[t+bars]/close[t] - 1.
    Recomputes the expected value and compares."""
    expected = (pl.col("close").shift(-bars).over("asset_id") / pl.col("close") - 1).alias(
        "_expected"
    )
    with_expected = df.with_columns(expected)
    diffs = (
        with_expected.filter(pl.col(label_name).is_not_null())
        .select((pl.col(label_name) - pl.col("_expected")).abs().max().alias("max_diff"))
        .item()
    )
    if diffs is None or diffs > tolerance:
        raise DatasetValidationError(
            f"label {label_name!r} does not match close[t+{bars}]/close[t] - 1 (max diff = {diffs})"
        )


def validate_temporal_dataset(
    train: DatasetSplit,
    val: DatasetSplit,
    test: DatasetSplit,
    feature_names: list[str],
    label_names: list[str],
    purge_bars: int,
    bar_delta: timedelta,
    label_bars: int,
    label_name: str,
) -> None:
    """Run all checks on a temporal split."""
    check_no_overlap(train, val, test)
    check_purge_gap(train, val, bar_delta, purge_bars)
    check_purge_gap(val, test, bar_delta, purge_bars)
    required = feature_names + label_names
    for split in (train, val, test):
        check_no_nulls(split, required)
        check_weights_sum(split)
        check_label_is_forward_return(split.df, label_name, label_bars)
