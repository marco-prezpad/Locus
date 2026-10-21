"""
test_strategies.py

Strategy structure: weight shapes, sum-to-1, per-rebalance cadence,
null feature handling, seeding.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.baselines.strategies import (
    BuyAndHoldBTC,
    EqualWeightUniverse,
    MomentumTopK,
    RandomTopK,
)
from locus.data.providers.base import BAR_SCHEMA


def _df(n: int = 100, assets: tuple[str, ...] = ("BTC", "ETH", "SOL")) -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for i, asset in enumerate(assets):
        for t in range(n):
            price = 100.0 + 10 * i + 0.1 * t
            rows.append(
                {
                    "asset_id": asset,
                    "venue": "test",
                    "symbol": f"{asset}USDT",
                    "timeframe": "1h",
                    "timestamp": base + timedelta(hours=t),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1.0,
                }
            )
    df = pl.DataFrame(rows, schema=BAR_SCHEMA)
    # Add a fake return_20 feature so momentum can run
    return df.with_columns(
        (pl.col("close") / pl.col("close").shift(20).over("asset_id") - 1).alias("return_20")
    )


def test_buy_and_hold_btc_weights() -> None:
    df = _df()
    w = BuyAndHoldBTC().generate_weights(df, rebalance_freq=24)
    assert w.height > 0
    assert set(w.columns) == {"timestamp", "asset_id", "weight"}
    # Sum per timestamp = 1
    sums = w.group_by("timestamp").agg(pl.col("weight").sum().alias("s"))
    assert (sums["s"] - 1.0).abs().max() < 1e-9


def test_equal_weight_sums_to_one() -> None:
    df = _df()
    w = EqualWeightUniverse().generate_weights(df, rebalance_freq=24)
    sums = w.group_by("timestamp").agg(pl.col("weight").sum().alias("s"))
    assert (sums["s"] - 1.0).abs().max() < 1e-9


def test_momentum_top_k_selects_k() -> None:
    df = _df()
    w = MomentumTopK(source="return_20", k=2).generate_weights(df, rebalance_freq=24)
    # After the warm-up, each timestamp should have exactly K non-zero
    # weights (or fewer at boundaries)
    nonzero = w.filter(pl.col("weight") > 0).group_by("timestamp").agg(pl.len().alias("n"))
    assert nonzero["n"].max() <= 2


def test_momentum_requires_source() -> None:
    df = _df().drop("return_20")
    with pytest.raises(ValueError, match="source column"):
        MomentumTopK(source="return_20", k=2).generate_weights(df, 24)


def test_random_top_k_reproducible() -> None:
    df = _df()
    w1 = RandomTopK(k=2, seed=42).generate_weights(df, rebalance_freq=24)
    w2 = RandomTopK(k=2, seed=42).generate_weights(df, rebalance_freq=24)
    assert w1.equals(w2)


def test_random_top_k_different_seeds() -> None:
    df = _df()
    w1 = RandomTopK(k=2, seed=42).generate_weights(df, rebalance_freq=24)
    w2 = RandomTopK(k=2, seed=99).generate_weights(df, rebalance_freq=24)
    assert not w1.equals(w2)


def test_momentum_rejects_zero_k() -> None:
    with pytest.raises(ValueError):
        MomentumTopK(k=0)
