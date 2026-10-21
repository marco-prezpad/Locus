"""
test_validation.py

DatasetValidator: no overlap, purge gap, no nulls, weights sum,
label recomputation.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.data.providers.base import BAR_SCHEMA
from locus.datasets import DatasetSplit
from locus.datasets.validation import (
    DatasetValidationError,
    check_label_is_forward_return,
    check_no_nulls,
    check_no_overlap,
    check_purge_gap,
    check_weights_sum,
    validate_temporal_dataset,
)


def _split(name: str, start_hr: int, end_hr: int) -> DatasetSplit:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    n = end_hr - start_hr + 1
    rows = [
        {
            "asset_id": "A",
            "venue": "t",
            "symbol": "AUSDT",
            "timeframe": "1h",
            "timestamp": base + timedelta(hours=start_hr + i),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1.0,
        }
        for i in range(n)
    ]
    df = pl.DataFrame(rows, schema=BAR_SCHEMA)
    return DatasetSplit(
        name=name,
        df=df,
        weights=pl.Series("uniqueness_weight", [1.0] * n, dtype=pl.Float64),
        n_eff=float(n),
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
    )


def test_no_overlap_passes() -> None:
    train = _split("train", 0, 100)
    val = _split("val", 200, 300)
    test = _split("test", 400, 500)
    check_no_overlap(train, val, test)


def test_overlap_raises() -> None:
    train = _split("train", 0, 250)
    val = _split("val", 200, 300)
    test = _split("test", 400, 500)
    with pytest.raises(DatasetValidationError, match="train end"):
        check_no_overlap(train, val, test)


def test_purge_gap_raises_when_too_small() -> None:
    train = _split("train", 0, 100)
    val = _split("val", 110, 200)  # gap = 10 hours, purge = 74
    with pytest.raises(DatasetValidationError, match="purge gap"):
        check_purge_gap(train, val, timedelta(hours=1), 74)


def test_no_nulls_passes() -> None:
    split = _split("train", 0, 10)
    check_no_nulls(split, ["close"])


def test_nulls_raise() -> None:
    split = _split("train", 0, 10)
    df_with_nulls = split.df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("close")).alias("close")
    )
    split_bad = DatasetSplit(
        name="train",
        df=df_with_nulls,
        weights=split.weights,
        n_eff=split.n_eff,
        start=split.start,
        end=split.end,
    )
    with pytest.raises(DatasetValidationError, match="nulls"):
        check_no_nulls(split_bad, ["close"])


def test_weights_sum_mismatch_raises() -> None:
    split = _split("train", 0, 10)
    bad = DatasetSplit(
        name="train",
        df=split.df,
        weights=pl.Series("uniqueness_weight", [0.5] * 10, dtype=pl.Float64),
        n_eff=10.0,  # but actual sum is 5.0
        start=split.start,
        end=split.end,
    )
    with pytest.raises(DatasetValidationError, match="n_eff mismatch"):
        check_weights_sum(bad)


def test_label_forward_return_recomputation() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {
            "asset_id": "A",
            "venue": "t",
            "symbol": "AUSDT",
            "timeframe": "1h",
            "timestamp": base + timedelta(hours=i),
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": 1.0,
        }
        for i, c in enumerate([100.0, 110.0, 121.0, 133.1])
    ]
    df = pl.DataFrame(rows, schema=BAR_SCHEMA).with_columns(
        (pl.col("close").shift(-1).over("asset_id") / pl.col("close") - 1).alias("future_return_1h")
    )
    check_label_is_forward_return(df, "future_return_1h", 1)


def test_val_test_overlap_raises() -> None:
    """check_no_overlap second branch: val.end >= test.start."""
    train = _split("train", 0, 100)
    val = _split("val", 200, 300)
    test = _split("test", 250, 350)  # overlaps val
    with pytest.raises(DatasetValidationError, match="val end"):
        check_no_overlap(train, val, test)


def test_val_test_purge_gap_raises() -> None:
    """check_purge_gap between val and test (not just train/val)."""
    val = _split("val", 0, 100)
    test = _split("test", 110, 200)  # gap = 10 hours, purge = 74
    with pytest.raises(DatasetValidationError, match="purge gap"):
        check_purge_gap(val, test, timedelta(hours=1), 74)


def test_label_mismatch_raises() -> None:
    """check_label_is_forward_return branch: diffs > tolerance."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    closes = [100.0, 110.0, 121.0, 133.1]
    rows = [
        {
            "asset_id": "A",
            "venue": "t",
            "symbol": "AUSDT",
            "timeframe": "1h",
            "timestamp": base + timedelta(hours=i),
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": 1.0,
        }
        for i, c in enumerate(closes)
    ]
    df = pl.DataFrame(rows, schema=BAR_SCHEMA).with_columns(
        # Deliberately wrong: +50% instead of the true +10%
        pl.lit(0.50).alias("future_return_1h")
    )
    with pytest.raises(DatasetValidationError, match="does not match"):
        check_label_is_forward_return(df, "future_return_1h", 1)


def test_label_all_null_raises() -> None:
    """check_label_is_forward_return branch: diffs is None (all labels
    null in the DataFrame)."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {
            "asset_id": "A",
            "venue": "t",
            "symbol": "AUSDT",
            "timeframe": "1h",
            "timestamp": base + timedelta(hours=i),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1.0,
        }
        for i in range(4)
    ]
    df = pl.DataFrame(rows, schema=BAR_SCHEMA).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("future_return_1h")
    )
    with pytest.raises(DatasetValidationError, match="does not match"):
        check_label_is_forward_return(df, "future_return_1h", 1)


def _make_clean_split(name: str, start_hr: int, end_hr: int) -> DatasetSplit:
    """Build a clean split with correct forward returns and no nulls.
    Used by the validate_temporal_dataset end-to-end tests.

    `volume` is used as the dummy feature column in the end-to-end
    tests because it is part of BAR_SCHEMA and always non-null here.
    """
    base = datetime(2024, 1, 1, tzinfo=UTC)
    n = end_hr - start_hr + 1
    closes = [100.0 + i for i in range(n)]
    rows = [
        {
            "asset_id": "A",
            "venue": "t",
            "symbol": "AUSDT",
            "timeframe": "1h",
            "timestamp": base + timedelta(hours=start_hr + i),
            "open": closes[i],
            "high": closes[i],
            "low": closes[i],
            "close": closes[i],
            "volume": 1.0,
        }
        for i in range(n)
    ]
    df = pl.DataFrame(rows, schema=BAR_SCHEMA).with_columns(
        (pl.col("close").shift(-1) / pl.col("close") - 1).alias("future_return_1h")
    )
    df = df.drop_nulls(subset=["future_return_1h"])
    m = df.height
    return DatasetSplit(
        name=name,
        df=df,
        weights=pl.Series("uniqueness_weight", [1.0] * m, dtype=pl.Float64),
        n_eff=float(m),
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
    )


def test_validate_temporal_dataset_passes_end_to_end() -> None:
    """Full validate_temporal_dataset on a valid split: exercises
    no_overlap (both branches), purge_gap (both branches), no_nulls,
    weights_sum, and label recomputation (passing branch)."""
    # train ends at hour 99; val starts at 180; gap = 81 > 74.
    # val ends at 279; test starts at 360; gap = 81 > 74.
    train = _make_clean_split("train", 0, 100)
    val = _make_clean_split("val", 180, 280)
    test = _make_clean_split("test", 360, 460)

    validate_temporal_dataset(
        train=train,
        val=val,
        test=test,
        feature_names=["volume"],
        label_names=["future_return_1h"],
        purge_bars=74,
        bar_delta=timedelta(hours=1),
        label_bars=1,
        label_name="future_return_1h",
    )


def test_validate_temporal_dataset_fails_on_small_train_val_purge() -> None:
    """validate_temporal_dataset propagates a purge-gap violation
    between train and val."""
    # train ends at 99; val starts at 110; gap = 11 < 74 → raises.
    train = _make_clean_split("train", 0, 100)
    val = _make_clean_split("val", 110, 210)
    test = _make_clean_split("test", 360, 460)

    with pytest.raises(DatasetValidationError, match="purge gap"):
        validate_temporal_dataset(
            train=train,
            val=val,
            test=test,
            feature_names=["volume"],
            label_names=["future_return_1h"],
            purge_bars=74,
            bar_delta=timedelta(hours=1),
            label_bars=1,
            label_name="future_return_1h",
        )


def test_validate_temporal_dataset_fails_on_small_val_test_purge() -> None:
    """validate_temporal_dataset propagates a purge-gap violation
    between val and test. This exercises the check_purge_gap(val, test)
    call inside validate_temporal_dataset."""
    # train->val gap OK (180-99=81). val->test gap bad (300-279=21).
    train = _make_clean_split("train", 0, 100)
    val = _make_clean_split("val", 180, 280)
    test = _make_clean_split("test", 300, 400)

    with pytest.raises(DatasetValidationError, match="purge gap"):
        validate_temporal_dataset(
            train=train,
            val=val,
            test=test,
            feature_names=["volume"],
            label_names=["future_return_1h"],
            purge_bars=74,
            bar_delta=timedelta(hours=1),
            label_bars=1,
            label_name="future_return_1h",
        )
