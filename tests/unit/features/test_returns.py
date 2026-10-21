"""
test_returns.py

ReturnFeature and LogReturnFeature: numerical correctness on a known
input, nulls for the initial rows, per-asset grouping.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.data.providers.base import BAR_SCHEMA
from locus.features import LogReturnFeature, ReturnFeature


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
    if closes_b is not None:
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


def test_return_1_numerically_correct() -> None:
    df = _df([100.0, 110.0, 121.0, 133.1])
    result = ReturnFeature(name="return_1", n=1).compute(df).sort("timestamp")
    values = result["return_1"].to_list()
    assert values[0] is None
    assert values[1] == pytest.approx(0.10)
    assert values[2] == pytest.approx(0.10)
    assert values[3] == pytest.approx(0.10)


def test_return_2_numerically_correct() -> None:
    df = _df([100.0, 110.0, 121.0, 133.1])
    result = ReturnFeature(name="return_2", n=2).compute(df).sort("timestamp")
    values = result["return_2"].to_list()
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(0.21)
    assert values[3] == pytest.approx(0.21)


def test_return_groups_by_asset() -> None:
    df = _df([100.0, 110.0], closes_b=[200.0, 220.0])
    result = ReturnFeature(name="return_1", n=1).compute(df)
    a_values = result.filter(pl.col("asset_id") == "A").sort("timestamp")["return_1"].to_list()
    b_values = result.filter(pl.col("asset_id") == "B").sort("timestamp")["return_1"].to_list()
    assert a_values[1] == pytest.approx(0.10)
    assert b_values[1] == pytest.approx(0.10)
    assert a_values[0] is None
    assert b_values[0] is None


def test_log_return_numerically_correct() -> None:
    df = _df([100.0, 110.0])
    result = LogReturnFeature(name="log_return_1", n=1).compute(df).sort("timestamp")
    values = result["log_return_1"].to_list()
    assert values[0] is None
    assert values[1] == pytest.approx(0.0953101798, rel=1e-8)


def test_return_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError):
        ReturnFeature(name="x", n=0)
