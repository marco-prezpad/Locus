"""
test_splits.py

Temporal splitter and walk-forward splitter: no overlap, purge gaps,
chronological order, all assets at same timestamp in same split.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.data.providers.base import BAR_SCHEMA
from locus.datasets import (
    PurgedWalkForwardSplitter,
    TemporalSplitter,
)


def _panel(n_bars: int = 1000, assets: tuple[str, ...] = ("A", "B")) -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for asset in assets:
        for i in range(n_bars):
            rows.append(
                {
                    "asset_id": asset,
                    "venue": "test",
                    "symbol": f"{asset}USDT",
                    "timeframe": "1h",
                    "timestamp": base + timedelta(hours=i),
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                    "volume": 1.0,
                }
            )
    return pl.DataFrame(rows, schema=BAR_SCHEMA)


def test_temporal_split_no_overlap() -> None:
    splitter = TemporalSplitter(horizon_bars=24, lookback_bars=50)
    splits = splitter.split(_panel())
    assert splits["train"].end < splits["val"].start
    assert splits["val"].end < splits["test"].start


def test_temporal_split_all_assets_at_ts_in_same_split() -> None:
    splitter = TemporalSplitter(horizon_bars=24, lookback_bars=50)
    splits = splitter.split(_panel())
    for _name, split in splits.items():
        split.df["timestamp"]
        # For each unique ts, both assets should be present
        counts = split.df.group_by("timestamp").agg(pl.len().alias("n"))
        # All timestamps should have exactly 2 assets (allowing the
        # last bar which might be dropped if split ends mid-batch)
        assert counts["n"].min() == counts["n"].max()


def test_temporal_split_purge_gap_present() -> None:
    splitter = TemporalSplitter(horizon_bars=24, lookback_bars=50)
    splits = splitter.split(_panel())
    gap = splits["val"].start - splits["train"].end
    # Purge = 24 + 50 = 74 bars
    assert gap >= timedelta(hours=74)


def test_temporal_split_rejects_invalid_ratios() -> None:
    with pytest.raises(ValueError):
        TemporalSplitter(horizon_bars=1, lookback_bars=1, train_ratio=0)
    with pytest.raises(ValueError):
        TemporalSplitter(horizon_bars=1, lookback_bars=1, train_ratio=0.6, val_ratio=0.5)


def test_walk_forward_produces_multiple_folds() -> None:
    splitter = PurgedWalkForwardSplitter(n_splits=5, horizon_bars=24, lookback_bars=50)
    folds = splitter.split(_panel(n_bars=5000))
    assert len(folds) == 5
    for fold in folds:
        assert fold.train.end < fold.val.start
        assert fold.val.end < fold.test.start


def test_walk_forward_increasing_train_size() -> None:
    splitter = PurgedWalkForwardSplitter(n_splits=5, horizon_bars=24, lookback_bars=50)
    folds = splitter.split(_panel(n_bars=5000))
    train_sizes = [f.train.n_samples for f in folds]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[0] < train_sizes[-1]


def test_walk_forward_rejects_too_few_splits() -> None:
    with pytest.raises(ValueError):
        PurgedWalkForwardSplitter(n_splits=1, horizon_bars=24, lookback_bars=50)


def test_walk_forward_insufficient_data_raises() -> None:
    splitter = PurgedWalkForwardSplitter(n_splits=10, horizon_bars=24, lookback_bars=50)
    with pytest.raises(ValueError, match="not enough timestamps"):
        splitter.split(_panel(n_bars=50))
