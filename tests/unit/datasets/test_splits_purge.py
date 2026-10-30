"""Purge and embargo verification for P4 (plan §9)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.datasets.splits import PurgedWalkForwardSplitter


def _df(n_hours: int = 2000, n_assets: int = 3) -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for asset_id in [f"A{i}" for i in range(n_assets)]:
        for h in range(n_hours):
            rows.append(
                {
                    "asset_id": asset_id,
                    "timestamp": base + timedelta(hours=h),
                    "open": 100.0,
                    "close": 100.0,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def _splitter(horizon: int = 24, n_splits: int = 5) -> PurgedWalkForwardSplitter:
    return PurgedWalkForwardSplitter(
        n_splits=n_splits,
        horizon_bars=horizon,
        lookback_bars=0,
        test_ratio=0.15,
    )


def test_five_folds_produced() -> None:
    folds = _splitter().split(_df())
    assert len(folds) == 5


def test_each_fold_has_three_splits() -> None:
    folds = _splitter().split(_df())
    for f in folds:
        assert f.train.n_samples > 0
        assert f.val.n_samples > 0
        assert f.test.n_samples > 0


def test_no_overlap_train_val() -> None:
    folds = _splitter().split(_df())
    for f in folds:
        overlap = set(f.train.df["timestamp"].to_list()) & set(f.val.df["timestamp"].to_list())
        assert not overlap, f"fold {f.fold}: {len(overlap)} overlapping train/val"


def test_no_overlap_val_test() -> None:
    folds = _splitter().split(_df())
    for f in folds:
        overlap = set(f.val.df["timestamp"].to_list()) & set(f.test.df["timestamp"].to_list())
        assert not overlap, f"fold {f.fold}: {len(overlap)} overlapping val/test"


def test_purge_gap_train_val_at_least_horizon() -> None:
    horizon = 24
    folds = _splitter(horizon=horizon).split(_df())
    for f in folds:
        train_max = f.train.df["timestamp"].max()
        val_min = f.val.df["timestamp"].min()
        gap = int((val_min - train_max).total_seconds() / 3600)
        assert gap >= horizon, f"fold {f.fold}: purge gap {gap}h < {horizon}h"


def test_embargo_gap_val_test_at_least_horizon() -> None:
    horizon = 24
    folds = _splitter(horizon=horizon).split(_df())
    for f in folds:
        val_max = f.val.df["timestamp"].max()
        test_min = f.test.df["timestamp"].min()
        gap = int((test_min - val_max).total_seconds() / 3600)
        assert gap >= horizon, f"fold {f.fold}: embargo gap {gap}h < {horizon}h"


def test_test_is_holdout_at_end() -> None:
    """The splitter uses a fixed test holdout at the end of the series."""
    df = _df()
    folds = _splitter().split(df)
    last_ts = df["timestamp"].max()
    first_test_ts = folds[0].test.df["timestamp"].min()
    # First fold and last fold share the same test start
    for f in folds:
        assert f.test.df["timestamp"].min() == first_test_ts
        assert f.test.df["timestamp"].max() == last_ts


def test_train_grows_across_folds() -> None:
    """Train size (in samples) increases with each fold."""
    folds = _splitter().split(_df())
    sizes = [f.train.n_samples for f in folds]
    assert sizes == sorted(sizes), f"train sizes not increasing: {sizes}"
    assert sizes[0] < sizes[-1]


def test_reproducible_same_input() -> None:
    df = _df()
    a = _splitter().split(df)
    b = _splitter().split(df)
    for fa, fb in zip(a, b, strict=True):
        assert fa.train.df.equals(fb.train.df)
        assert fa.val.df.equals(fb.val.df)
        assert fa.test.df.equals(fb.test.df)


def test_not_enough_history_raises() -> None:
    """Very short series cannot produce n_splits folds."""
    with pytest.raises(ValueError, match="not enough timestamps"):
        _splitter(n_splits=20).split(_df(n_hours=100))
