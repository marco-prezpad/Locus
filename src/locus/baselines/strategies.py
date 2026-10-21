"""
strategies.py

Four naive baselines: Buy&Hold BTC, EqualWeight across the universe,
MomentumTopK by return_20, and RandomTopK (seeded).

All return weights summing to 1 at each rebalance. Zero-weight rows are
included for assets that are available but not selected; this makes the
turnover and join logic cleaner downstream.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


def _rebalance_timestamps(df: pl.DataFrame, rebalance_freq: int) -> pl.Series:
    """Unique timestamps at multiples of `rebalance_freq` bars from the
    first timestamp."""
    ts = df["timestamp"].unique().sort()
    if ts.len() == 0:
        return ts
    n_reb = (ts.len() + rebalance_freq - 1) // rebalance_freq
    indices = [min(i * rebalance_freq, ts.len() - 1) for i in range(n_reb)]
    return ts.gather(indices)


def _weights_at_rebalances(
    df: pl.DataFrame,
    rebalance_ts: pl.Series,
    weight_fn,
) -> pl.DataFrame:
    """For each rebalance timestamp, apply `weight_fn(group)` (a
    DataFrame → DataFrame with asset_id, weight) and concat."""
    out = []
    reb_set = set(rebalance_ts.to_list())
    rebalance_df = df.filter(pl.col("timestamp").is_in(list(reb_set)))
    for (ts,), group in rebalance_df.group_by("timestamp", maintain_order=True):
        w = weight_fn(group)
        out.append(w.with_columns(pl.lit(ts).alias("timestamp")))
    if not out:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "asset_id": pl.Utf8,
                "weight": pl.Float64,
            }
        )
    result = pl.concat(out).select(["timestamp", "asset_id", "weight"])
    return result.sort(["timestamp", "asset_id"])


@dataclass(frozen=True, slots=True)
class BuyAndHoldBTC:
    """Weight 1 on BTC, 0 on everything else."""

    _name: str = "buy_and_hold_btc"
    btc_asset_id: str = "BTC"

    @property
    def name(self) -> str:
        return self._name

    def generate_weights(self, df: pl.DataFrame, rebalance_freq: int) -> pl.DataFrame:
        reb = _rebalance_timestamps(df, rebalance_freq)

        def fn(group: pl.DataFrame) -> pl.DataFrame:
            return group.select("asset_id").with_columns(
                pl.when(pl.col("asset_id") == self.btc_asset_id)
                .then(pl.lit(1.0))
                .otherwise(pl.lit(0.0))
                .alias("weight")
            )

        return _weights_at_rebalances(df, reb, fn)


@dataclass(frozen=True, slots=True)
class EqualWeightUniverse:
    """Weight 1/N on each of the N assets available at each rebalance."""

    _name: str = "equal_weight"

    @property
    def name(self) -> str:
        return self._name

    def generate_weights(self, df: pl.DataFrame, rebalance_freq: int) -> pl.DataFrame:
        reb = _rebalance_timestamps(df, rebalance_freq)

        def fn(group: pl.DataFrame) -> pl.DataFrame:
            n = group.height
            if n == 0:
                return group.select("asset_id").with_columns(pl.lit(0.0).alias("weight"))
            return group.select("asset_id").with_columns(pl.lit(1.0 / n).alias("weight"))

        return _weights_at_rebalances(df, reb, fn)


@dataclass(frozen=True, slots=True)
class MomentumTopK:
    """Top K assets by `source` feature at each rebalance; equal-weight
    within the K. Assets with null source are excluded."""

    source: str = "return_20"
    k: int = 3
    _name: str = "momentum_top_k"

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be positive")

    @property
    def name(self) -> str:
        return f"momentum_top_{self.k}"

    def generate_weights(self, df: pl.DataFrame, rebalance_freq: int) -> pl.DataFrame:
        if self.source not in df.columns:
            raise ValueError(f"source column {self.source!r} not in df")
        # Only rebalance at timestamps where at least K assets have
        # non-null source
        candidates = df.filter(pl.col(self.source).is_not_null())
        reb = _rebalance_timestamps(candidates, rebalance_freq)

        def fn(group: pl.DataFrame) -> pl.DataFrame:
            group_clean = group.filter(pl.col(self.source).is_not_null())
            if group_clean.height == 0:
                return group.select("asset_id").with_columns(pl.lit(0.0).alias("weight"))
            top = group_clean.sort(self.source, descending=True).head(self.k)
            top_ids = set(top["asset_id"].to_list())
            weight = 1.0 / top.height
            return group.select("asset_id").with_columns(
                pl.when(pl.col("asset_id").is_in(list(top_ids)))
                .then(pl.lit(weight))
                .otherwise(pl.lit(0.0))
                .alias("weight")
            )

        return _weights_at_rebalances(df, reb, fn)


@dataclass(frozen=True, slots=True)
class RandomTopK:
    """Random selection of K assets at each rebalance; equal-weight
    within the K. Seeded for reproducibility."""

    k: int = 3
    seed: int = 42
    _name: str = "random_top_k"

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be positive")

    @property
    def name(self) -> str:
        return f"random_top_{self.k}"

    def generate_weights(self, df: pl.DataFrame, rebalance_freq: int) -> pl.DataFrame:
        reb = _rebalance_timestamps(df, rebalance_freq)
        rng = np.random.default_rng(self.seed)

        def fn(group: pl.DataFrame) -> pl.DataFrame:
            n = group.height
            if n == 0:
                return group.select("asset_id").with_columns(pl.lit(0.0).alias("weight"))
            k = min(self.k, n)
            chosen = rng.choice(n, size=k, replace=False)
            chosen_ids = {group["asset_id"][int(i)] for i in chosen}
            weight = 1.0 / k
            return group.select("asset_id").with_columns(
                pl.when(pl.col("asset_id").is_in(list(chosen_ids)))
                .then(pl.lit(weight))
                .otherwise(pl.lit(0.0))
                .alias("weight")
            )

        return _weights_at_rebalances(df, reb, fn)
