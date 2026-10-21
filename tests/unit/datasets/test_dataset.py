"""
test_dataset.py

TrainingDatasetBuilder: full pipeline (features → labels → drop nulls
→ split), dispatch to temporal vs walk-forward, missing column errors.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.data.providers.base import BAR_SCHEMA
from locus.datasets import (
    LabelBuilder,
    LabelSpec,
    PurgedWalkForwardSplitter,
    TemporalDataset,
    TemporalSplitter,
    TrainingDatasetBuilder,
    WalkForwardDataset,
)
from locus.features import FeaturePipeline, FeatureSpec


def _ohlcv(
    n: int = 300,
    assets: tuple[str, ...] = ("BTC", "ETH"),
) -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for i, asset in enumerate(assets):
        for t in range(n):
            price = 100.0 * (1 + 0.001 * (i + 1)) ** t
            rows.append(
                {
                    "asset_id": asset,
                    "venue": "test",
                    "symbol": f"{asset}USDT",
                    "timeframe": "1h",
                    "timestamp": base + timedelta(hours=t),
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1.0 + 0.1 * t,
                }
            )
    return pl.DataFrame(rows, schema=BAR_SCHEMA)


def _pipeline() -> FeaturePipeline:
    return FeaturePipeline.from_specs(
        [
            FeatureSpec(name="return_1", family="return", params={"n": 1}),
            FeatureSpec(name="return_5", family="return", params={"n": 5}),
            FeatureSpec(name="return_20", family="return", params={"n": 20}),
        ]
    )


def _labels() -> LabelBuilder:
    return LabelBuilder(
        [
            LabelSpec(name="future_return_1h", bars=1),
            LabelSpec(name="future_return_24h", bars=24),
        ],
        probability_positive_from="future_return_24h",
    )


def test_build_from_ohlcv_temporal() -> None:
    splitter = TemporalSplitter(horizon_bars=24, lookback_bars=50)
    builder = TrainingDatasetBuilder(_pipeline(), _labels(), splitter)
    result = builder.build_from_ohlcv(_ohlcv())
    assert isinstance(result, TemporalDataset)
    assert result.feature_names == ["return_1", "return_5", "return_20"]
    assert "future_return_24h" in result.label_names
    assert "probability_positive" in result.label_names


def test_build_from_ohlcv_drops_null_rows() -> None:
    # Use a larger input so the warm-up (~21 bars per asset) and the
    # purge (~74 bars per boundary) are a small fraction of the total.
    # With 300 bars, the purge alone eats ~50% and the test threshold
    # would be unachievable regardless of implementation.
    splitter = TemporalSplitter(horizon_bars=24, lookback_bars=50)
    builder = TrainingDatasetBuilder(_pipeline(), _labels(), splitter)
    df = _ohlcv(n=2000)
    result = builder.build_from_ohlcv(df)
    total = result.train.n_samples + result.val.n_samples + result.test.n_samples
    input_total = df.height
    # Warm-up (~21 bars/asset) + labels end (~24 bars/asset) + 2×purge
    # (74 bars per boundary) removed. With 2000 bars/asset, that's
    # ~10% loss, not 50%.
    assert total < input_total
    assert total > input_total * 0.75


def test_build_from_ohlcv_small_input_loses_more_to_purge() -> None:
    """Documents that small inputs lose a large fraction to the purge.
    This is expected: purge is absolute (74 bars), not relative."""
    splitter = TemporalSplitter(horizon_bars=24, lookback_bars=50)
    builder = TrainingDatasetBuilder(_pipeline(), _labels(), splitter)
    df = _ohlcv(n=300)
    result = builder.build_from_ohlcv(df)
    total = result.train.n_samples + result.val.n_samples + result.test.n_samples
    # Small inputs lose >30% to warm-up + purge
    assert total < df.height * 0.6


def test_build_from_ohlcv_walk_forward() -> None:
    splitter = PurgedWalkForwardSplitter(n_splits=3, horizon_bars=24, lookback_bars=50)
    builder = TrainingDatasetBuilder(_pipeline(), _labels(), splitter)
    result = builder.build_from_ohlcv(_ohlcv(n=1000))
    assert isinstance(result, WalkForwardDataset)
    assert len(result.folds) == 3
    for fold in result.folds:
        assert fold.train.n_samples > 0
        assert fold.val.n_samples > 0


def test_build_from_labelled_requires_columns() -> None:
    splitter = TemporalSplitter(horizon_bars=24, lookback_bars=50)
    builder = TrainingDatasetBuilder(_pipeline(), _labels(), splitter)
    # Missing all feature and label columns
    with pytest.raises(ValueError, match="missing columns"):
        builder.build_from_labelled(_ohlcv())


def test_build_from_labelled_succeeds_with_prebuilt_df() -> None:
    pipeline = _pipeline()
    labels = _labels()
    # Build the df as the pipeline would
    with_features = pipeline.compute(_ohlcv())
    with_labels = labels.compute(with_features)
    cleaned = with_labels.drop_nulls(subset=pipeline.feature_names + labels.label_names)
    splitter = TemporalSplitter(horizon_bars=24, lookback_bars=50)
    builder = TrainingDatasetBuilder(pipeline, labels, splitter)
    result = builder.build_from_labelled(cleaned)
    assert isinstance(result, TemporalDataset)


def test_build_from_labelled_also_drops_nulls() -> None:
    """Even if the input has nulls, build_from_labelled drops them."""
    pipeline = _pipeline()
    labels = _labels()
    with_features = pipeline.compute(_ohlcv())
    with_labels = labels.compute(with_features)
    # Don't pre-drop; feed with nulls
    splitter = TemporalSplitter(horizon_bars=24, lookback_bars=50)
    builder = TrainingDatasetBuilder(pipeline, labels, splitter)
    result = builder.build_from_labelled(with_labels)
    assert isinstance(result, TemporalDataset)
    # No nulls in any split
    for split in result.all_splits:
        for col in pipeline.feature_names + labels.label_names:
            assert split.df[col].null_count() == 0


def test_temporal_dataset_all_splits_property() -> None:
    splitter = TemporalSplitter(horizon_bars=24, lookback_bars=50)
    builder = TrainingDatasetBuilder(_pipeline(), _labels(), splitter)
    result = builder.build_from_ohlcv(_ohlcv())
    assert isinstance(result, TemporalDataset)
    splits = result.all_splits
    assert len(splits) == 3
    assert splits[0].name == "train"
    assert splits[1].name == "val"
    assert splits[2].name == "test"


def test_unknown_splitter_type_raises() -> None:
    splitter = TemporalSplitter(horizon_bars=24, lookback_bars=50)
    builder = TrainingDatasetBuilder(_pipeline(), _labels(), splitter)
    # Force the type check path by monkeypatching
    builder._splitter = object()  # type: ignore[assignment]
    with pytest.raises(TypeError, match="unknown splitter"):
        builder._apply_split(_ohlcv())
