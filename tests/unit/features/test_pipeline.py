"""
test_pipeline.py

FeaturePipeline end-to-end: column addition, ordering, idempotency,
empty input.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from locus.data.providers.base import BAR_SCHEMA
from locus.features import (
    FeaturePipeline,
    FeatureSpec,
    MovingAverageDistanceFeature,
    ReturnFeature,
)


def _df() -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "asset_id": "A",
                "venue": "test",
                "symbol": "AUSDT",
                "timeframe": "1h",
                "timestamp": base + timedelta(hours=i),
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.0 + i,
                "volume": 1.0,
            }
            for i in range(20)
        ],
        schema=BAR_SCHEMA,
    )


def test_pipeline_adds_one_column_per_feature() -> None:
    pipeline = FeaturePipeline(
        [
            ReturnFeature(name="return_1", n=1),
            MovingAverageDistanceFeature(name="ma_distance_5", window=5),
        ]
    )
    result = pipeline.compute(_df())
    assert result.width == _df().width + 2
    assert set(pipeline.feature_names) <= set(result.columns)


def test_pipeline_output_sorted() -> None:
    pipeline = FeaturePipeline([ReturnFeature(name="return_1", n=1)])
    result = pipeline.compute(_df().sample(fraction=1.0, shuffle=True))
    ts = result["timestamp"].to_list()
    assert ts == sorted(ts)


def test_pipeline_empty_input_returns_empty() -> None:
    pipeline = FeaturePipeline([ReturnFeature(name="return_1", n=1)])
    empty = pl.DataFrame(schema=BAR_SCHEMA)
    result = pipeline.compute(empty)
    assert result.height == 0


def test_pipeline_idempotent() -> None:
    pipeline = FeaturePipeline([ReturnFeature(name="return_1", n=1)])
    once = pipeline.compute(_df())
    twice = pipeline.compute(once)
    assert once["return_1"].equals(twice["return_1"])


def test_pipeline_from_specs() -> None:
    specs = [
        FeatureSpec(name="return_1", family="return", params={"n": 1}),
        FeatureSpec(name="ma_distance_5", family="ma_distance", params={"window": 5}),
    ]
    pipeline = FeaturePipeline.from_specs(specs)
    result = pipeline.compute(_df())
    assert "return_1" in result.columns
    assert "ma_distance_5" in result.columns
