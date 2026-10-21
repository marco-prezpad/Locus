"""
test_cross_sectional.py

Cross-sectional features: percentile rank, z-score, relative
momentum. Numerical correctness on synthetic multi-asset panels.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from locus.features import (
    CrossSectionalPercentileRankFeature,
    CrossSectionalZScoreFeature,
    RelativeMomentumVsMarketFeature,
)


def _panel(returns_per_asset: dict[str, float]) -> pl.DataFrame:
    """One timestamp, one row per asset, with `return_20` as a
    pre-populated column (simulating a prior feature)."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {
            "asset_id": asset_id,
            "venue": "test",
            "symbol": f"{asset_id}USDT",
            "timeframe": "1h",
            "timestamp": base,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "return_20": ret,
        }
        for asset_id, ret in returns_per_asset.items()
    ]
    return pl.DataFrame(rows).cast({"timestamp": pl.Datetime("us", "UTC")})


def test_percentile_rank_order() -> None:
    df = _panel({"A": 0.10, "B": 0.05, "C": 0.20})
    result = CrossSectionalPercentileRankFeature(name="xs_rank", source="return_20").compute(df)
    ranked = result.sort("asset_id")[["asset_id", "xs_rank"]]
    # A=0.10 → middle rank, B=0.05 → lowest, C=0.20 → highest
    b_val = float(ranked.filter(pl.col("asset_id") == "B")["xs_rank"][0])
    a_val = float(ranked.filter(pl.col("asset_id") == "A")["xs_rank"][0])
    c_val = float(ranked.filter(pl.col("asset_id") == "C")["xs_rank"][0])
    assert b_val < a_val < c_val


def test_percentile_rank_ties_give_same_value() -> None:
    df = _panel({"A": 0.10, "B": 0.10, "C": 0.20})
    result = CrossSectionalPercentileRankFeature(name="xs_rank", source="return_20").compute(df)
    a_val = float(result.filter(pl.col("asset_id") == "A")["xs_rank"][0])
    b_val = float(result.filter(pl.col("asset_id") == "B")["xs_rank"][0])
    assert a_val == pytest.approx(b_val)


def test_percentile_rank_source_missing_raises() -> None:
    df = _panel({"A": 0.10})
    with pytest.raises(ValueError, match="requires column"):
        CrossSectionalPercentileRankFeature(name="xs_rank", source="does_not_exist").compute(df)


def test_zscore_positive_for_above_mean() -> None:
    df = _panel({"A": 0.10, "B": 0.05, "C": 0.20})
    result = CrossSectionalZScoreFeature(name="xs_z", source="return_20").compute(df)
    a_z = float(result.filter(pl.col("asset_id") == "A")["xs_z"][0])
    b_z = float(result.filter(pl.col("asset_id") == "B")["xs_z"][0])
    c_z = float(result.filter(pl.col("asset_id") == "C")["xs_z"][0])
    # mean = 0.1167; A=0.10 below mean, B=0.05 below, C=0.20 above
    assert a_z < 0 < c_z
    assert b_z < a_z


def test_zscore_null_on_single_asset() -> None:
    df = _panel({"A": 0.10})
    result = CrossSectionalZScoreFeature(name="xs_z", source="return_20").compute(df)
    assert result["xs_z"][0] is None


def test_zscore_null_on_all_tied() -> None:
    df = _panel({"A": 0.10, "B": 0.10, "C": 0.10})
    result = CrossSectionalZScoreFeature(name="xs_z", source="return_20").compute(df)
    assert all(v is None for v in result["xs_z"].to_list())


def test_relative_momentum_sign() -> None:
    df = _panel({"A": 0.10, "B": 0.05, "C": 0.20})
    result = RelativeMomentumVsMarketFeature(name="rel_mom", source="return_20").compute(df)
    c_val = float(result.filter(pl.col("asset_id") == "C")["rel_mom"][0])
    b_val = float(result.filter(pl.col("asset_id") == "B")["rel_mom"][0])
    assert c_val > 0  # above market mean
    assert b_val < 0  # below market mean


def test_source_equals_name_raises() -> None:
    with pytest.raises(ValueError):
        CrossSectionalPercentileRankFeature(name="return_20", source="return_20")
