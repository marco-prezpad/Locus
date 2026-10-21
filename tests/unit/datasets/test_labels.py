"""
test_labels.py

LabelBuilder: numerical correctness, per-asset grouping, nulls at the
end, binary positive label.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.data.providers.base import BAR_SCHEMA
from locus.datasets import LabelBuilder, LabelSpec


def _df(closes_a: list[float], closes_b: list[float] | None = None) -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {
            "asset_id": "A",
            "venue": "test",
            "symbol": "AUSDT",
            "timeframe": "1h",
            "timestamp": base + timedelta(hours=i),
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": 1.0,
        }
        for i, c in enumerate(closes_a)
    ]
    if closes_b:
        rows += [
            {
                "asset_id": "B",
                "venue": "test",
                "symbol": "BUSDT",
                "timeframe": "1h",
                "timestamp": base + timedelta(hours=i),
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "volume": 1.0,
            }
            for i, c in enumerate(closes_b)
        ]
    return pl.DataFrame(rows, schema=BAR_SCHEMA)


def test_future_return_numerically_correct() -> None:
    df = _df([100.0, 110.0, 121.0, 133.1, 146.41])
    builder = LabelBuilder([LabelSpec(name="future_return_1h", bars=1)])
    result = builder.compute(df).sort("timestamp")
    values = result["future_return_1h"].to_list()
    assert values[0] == pytest.approx(0.10)
    assert values[1] == pytest.approx(0.10)
    assert values[3] == pytest.approx(0.10)
    assert values[-1] is None  # last bar has no future


def test_future_return_per_asset() -> None:
    df = _df([100.0, 110.0, 121.0], closes_b=[200.0, 220.0, 242.0])
    builder = LabelBuilder([LabelSpec(name="r", bars=1)])
    result = builder.compute(df)
    a = result.filter(pl.col("asset_id") == "A").sort("timestamp")["r"].to_list()
    b = result.filter(pl.col("asset_id") == "B").sort("timestamp")["r"].to_list()
    assert a[0] == pytest.approx(0.10)
    assert b[0] == pytest.approx(0.10)
    assert a[-1] is None
    assert b[-1] is None


def test_probability_positive_binary() -> None:
    df = _df([100.0, 110.0, 105.0, 115.0])
    builder = LabelBuilder(
        [LabelSpec(name="future_return_1h", bars=1)],
        probability_positive_from="future_return_1h",
    )
    result = builder.compute(df).sort("timestamp")
    values = result["probability_positive"].to_list()
    assert values[0] == 1  # 100 -> 110
    assert values[1] == 0  # 110 -> 105
    assert values[2] == 1  # 105 -> 115
    assert values[3] is None  # no future


def test_missing_probability_source_raises() -> None:
    with pytest.raises(ValueError, match="probability_positive_from"):
        LabelBuilder(
            [LabelSpec(name="r1", bars=1)],
            probability_positive_from="r2",
        )


def test_duplicate_label_names_raises() -> None:
    with pytest.raises(ValueError, match="duplicate label names"):
        LabelBuilder([LabelSpec(name="r", bars=1), LabelSpec(name="r", bars=4)])
