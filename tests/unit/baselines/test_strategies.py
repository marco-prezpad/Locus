"""
test_strategies.py

Strategy structure: weight shapes, sum-to-1, per-rebalance cadence,
null feature handling, seeding, MomentumTrend long-only logic.

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
    MomentumTrend,
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


def _df_from_prices(prices: dict[str, list[float]]) -> pl.DataFrame:
    """Build a BAR_SCHEMA DataFrame from an explicit price series per asset.

    Needed by MomentumTrend tests, where the *shape* of each asset's
    series matters (up / down / mixed), not just its level.
    """
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for asset, series in prices.items():
        for t, price in enumerate(series):
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
    return pl.DataFrame(rows, schema=BAR_SCHEMA)


# ---------------------------------------------------------------------------
# Buy-and-hold / equal-weight
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MomentumTopK
# ---------------------------------------------------------------------------


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


def test_momentum_rejects_zero_k() -> None:
    with pytest.raises(ValueError):
        MomentumTopK(k=0)


# ---------------------------------------------------------------------------
# RandomTopK
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MomentumTrend (long-only conditional)
# ---------------------------------------------------------------------------


def test_momentum_trend_long_when_price_above_ma() -> None:
    # Monotonic increasing => close > MA for every bar after warm-up.
    prices = {f"A{i}": [100.0 + t for t in range(50)] for i in range(3)}
    df = _df_from_prices(prices)
    w = MomentumTrend(window=10).generate_weights(df, rebalance_freq=24)

    # At the last rebalance, all three assets should be long.
    last = w.filter(pl.col("timestamp") == w["timestamp"].max())
    assert last.filter(pl.col("weight") > 0).height == 3
    assert (last["weight"] - 1 / 3).abs().max() < 1e-9


def test_momentum_trend_cash_when_price_below_ma() -> None:
    # Monotonic decreasing => close < MA => all weights 0.
    prices = {f"A{i}": [100.0 - t for t in range(50)] for i in range(3)}
    df = _df_from_prices(prices)
    w = MomentumTrend(window=10).generate_weights(df, rebalance_freq=24)
    assert (w["weight"] == 0.0).all()


def test_momentum_trend_mixed_signals_normalise_to_one() -> None:
    # A0 rises, A1 falls. Only A0 should be long, with weight 1.0.
    prices = {
        "A0": [100.0 + t for t in range(50)],
        "A1": [100.0 - t for t in range(50)],
    }
    df = _df_from_prices(prices)
    w = MomentumTrend(window=10).generate_weights(df, rebalance_freq=24)
    last = w.filter(pl.col("timestamp") == w["timestamp"].max())
    a0 = last.filter(pl.col("asset_id") == "A0")["weight"].item()
    a1 = last.filter(pl.col("asset_id") == "A1")["weight"].item()
    assert a0 == pytest.approx(1.0)
    assert a1 == pytest.approx(0.0)


def test_momentum_trend_weights_sum_le_one() -> None:
    # With 5 rising assets all long, sum = 1.0. With partial participation,
    # sum = n_active / n_active = 1.0 as well; <= 1 is the invariant.
    prices = {f"A{i}": [100.0 + t for t in range(50)] for i in range(5)}
    df = _df_from_prices(prices)
    w = MomentumTrend(window=10).generate_weights(df, rebalance_freq=24)
    sums = w.group_by("timestamp").agg(pl.col("weight").sum().alias("s"))
    assert (sums["s"] <= 1.0 + 1e-9).all()


def test_momentum_trend_no_look_ahead() -> None:
    # Changing the future must not change weights at earlier timestamps.
    base_prices = {f"A{i}": [100.0 + t for t in range(50)] for i in range(3)}
    df_base = _df_from_prices(base_prices)

    # Same prefix, different future (last 10 bars are negative).
    modified = {k: list(v) for k, v in base_prices.items()}
    for k in modified:
        modified[k][-10:] = [modified[k][-10] - t for t in range(10)]
    df_mod = _df_from_prices(modified)

    w_base = MomentumTrend(window=10).generate_weights(df_base, rebalance_freq=24)
    w_mod = MomentumTrend(window=10).generate_weights(df_mod, rebalance_freq=24)

    # Cutoff: last bar strictly before the modified future.
    cutoff = df_base["timestamp"].unique().sort()[-11]
    before_base = w_base.filter(pl.col("timestamp") <= cutoff)
    before_mod = w_mod.filter(pl.col("timestamp") <= cutoff)
    assert before_base.equals(before_mod)


def test_momentum_trend_rejects_zero_window() -> None:
    with pytest.raises(ValueError):
        MomentumTrend(window=0)
