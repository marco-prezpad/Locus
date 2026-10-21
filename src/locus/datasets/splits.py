"""
splits.py

Temporal and purged walk-forward splitters. Both preserve chronological
order and all assets at a given timestamp go to the same split.

Purge policy: `horizon_bars + lookback_bars`. Rationale:
- `horizon_bars`: prevents train label intervals [t, t+h] from
  overlapping val/test feature intervals.
- `lookback_bars`: prevents val/test features [t-lookback, t] from
  touching train label intervals [t, t+h] at t near the boundary.
The extended purge is more conservative than the standard `h` purge
and matches the plan §8.4.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from locus.datasets.types import DatasetSplit, WalkForwardFold
from locus.datasets.uniqueness import compute_uniqueness_weights
from locus.infrastructure.logging import LoggerFactory


def _infer_bar_delta(df: pl.DataFrame) -> timedelta:
    """Infer the bar spacing from the median gap of unique timestamps.

    Timestamps are `Datetime("us", "UTC")`; polars' `.dt.total_milliseconds()`
    gives an unambiguous unit regardless of the underlying storage. Using
    Int64 directly would return microseconds and silently produce 1-bar
    purges.
    """
    ts = df["timestamp"].unique().sort()
    if ts.len() < 2:
        return timedelta(hours=1)
    diffs_ms = ts.diff().drop_nulls().dt.total_milliseconds()
    if diffs_ms.len() == 0:
        return timedelta(hours=1)
    median_raw = diffs_ms.median()
    if not isinstance(median_raw, (int, float)):
        return timedelta(hours=1)
    median_ms = float(median_raw)
    if median_ms <= 0:
        return timedelta(hours=1)
    return timedelta(milliseconds=median_ms)


def _make_split(name: str, df: pl.DataFrame, horizon_bars: int) -> DatasetSplit:
    if df.height == 0:
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        return DatasetSplit(
            name=name,
            df=df,
            weights=pl.Series("uniqueness_weight", [], dtype=pl.Float64),
            n_eff=0.0,
            start=epoch,
            end=epoch,
        )
    weighted = compute_uniqueness_weights(df, horizon_bars)
    weights = weighted["uniqueness_weight"]
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    assert isinstance(ts_min, datetime)
    assert isinstance(ts_max, datetime)
    return DatasetSplit(
        name=name,
        df=weighted.drop("uniqueness_weight"),
        weights=weights,
        n_eff=float(weights.sum()),
        start=ts_min,
        end=ts_max,
    )


class TemporalSplitter:
    """Single train/val/test split by timestamp with purge gaps."""

    def __init__(
        self,
        horizon_bars: int,
        lookback_bars: int,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> None:
        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio must be in (0, 1)")
        if not (0 < val_ratio < 1):
            raise ValueError("val_ratio must be in (0, 1)")
        if train_ratio + val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be < 1")
        self._horizon = horizon_bars
        self._lookback = lookback_bars
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._log = LoggerFactory.get("datasets.splits.temporal")

    def split(self, df: pl.DataFrame) -> dict[str, DatasetSplit]:
        if df.height == 0:
            raise ValueError("cannot split an empty DataFrame")

        bar_dt = _infer_bar_delta(df)
        purge = bar_dt * (self._horizon + self._lookback)

        unique_ts = df["timestamp"].unique().sort()
        n = unique_ts.len()
        train_boundary = unique_ts[int(n * self._train_ratio)]
        val_boundary = unique_ts[int(n * (self._train_ratio + self._val_ratio))]

        train_end = train_boundary - purge
        val_start = train_boundary
        val_end = val_boundary - purge
        test_start = val_boundary

        train_df = df.filter(pl.col("timestamp") <= train_end)
        val_df = df.filter((pl.col("timestamp") >= val_start) & (pl.col("timestamp") <= val_end))
        test_df = df.filter(pl.col("timestamp") >= test_start)

        result = {
            "train": _make_split("train", train_df, self._horizon),
            "val": _make_split("val", val_df, self._horizon),
            "test": _make_split("test", test_df, self._horizon),
        }
        self._log.info(
            "temporal_split_done",
            extra={
                "train_n": result["train"].n_samples,
                "val_n": result["val"].n_samples,
                "test_n": result["test"].n_samples,
                "train_eff": round(result["train"].n_eff, 1),
                "val_eff": round(result["val"].n_eff, 1),
                "test_eff": round(result["test"].n_eff, 1),
                "purge_bars": self._horizon + self._lookback,
            },
        )
        return result


class PurgedWalkForwardSplitter:
    """K-fold walk-forward with purge gaps and a fixed test window."""

    def __init__(
        self,
        n_splits: int,
        horizon_bars: int,
        lookback_bars: int,
        test_ratio: float = 0.15,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not (0 < test_ratio < 0.5):
            raise ValueError("test_ratio must be in (0, 0.5)")
        self._n_splits = n_splits
        self._horizon = horizon_bars
        self._lookback = lookback_bars
        self._test_ratio = test_ratio
        self._log = LoggerFactory.get("datasets.splits.walk_forward")

    def split(self, df: pl.DataFrame) -> list[WalkForwardFold]:
        if df.height == 0:
            raise ValueError("cannot split an empty DataFrame")

        bar_dt = _infer_bar_delta(df)
        purge = bar_dt * (self._horizon + self._lookback)
        unique_ts = df["timestamp"].unique().sort()
        n = unique_ts.len()

        test_n = int(n * self._test_ratio)
        walkable = n - test_n
        # Each fold consumes one `fold_size` for val and one for the
        # initial train. n_splits folds require (n_splits + 1) windows.
        fold_size = walkable // (self._n_splits + 1)
        min_fold = self._horizon + self._lookback
        if fold_size < min_fold:
            raise ValueError(
                f"not enough timestamps ({n}) for {self._n_splits} folds: "
                f"fold_size={fold_size} < min_fold={min_fold} "
                f"(horizon+lookback); add more history"
            )

        folds: list[WalkForwardFold] = []
        for k in range(self._n_splits):
            train_end_idx = fold_size * (k + 1)
            val_end_idx = train_end_idx + fold_size
            if val_end_idx > walkable:
                break

            train_end = unique_ts[train_end_idx - 1] - purge
            val_start = unique_ts[train_end_idx]
            val_end = unique_ts[val_end_idx - 1] - purge
            test_start = unique_ts[walkable]
            test_end = unique_ts[n - 1]

            train_df = df.filter(pl.col("timestamp") <= train_end)
            val_df = df.filter(
                (pl.col("timestamp") >= val_start) & (pl.col("timestamp") <= val_end)
            )
            test_df = df.filter(
                (pl.col("timestamp") >= test_start) & (pl.col("timestamp") <= test_end)
            )
            if train_df.height == 0 or val_df.height == 0:
                continue

            folds.append(
                WalkForwardFold(
                    fold=k,
                    train=_make_split("train", train_df, self._horizon),
                    val=_make_split("val", val_df, self._horizon),
                    test=_make_split("test", test_df, self._horizon),
                )
            )

        self._log.info(
            "walk_forward_split_done",
            extra={
                "n_folds": len(folds),
                "test_n": folds[0].test.n_samples if folds else 0,
                "purge_bars": self._horizon + self._lookback,
            },
        )
        return folds
