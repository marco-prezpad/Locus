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


class FundingLongShortTopK:
    """Market-neutral funding strategy: long low funding, short high funding.

    At each rebalance timestamp t:
    - Filter to assets with non-null `feature` at t (warm-up excluded).
    - Rank by `feature` ascending: lowest funding first.
    - Long = top k (lowest funding z-score), short = bottom k (highest).
    - Weights: +gross/(2*k_eff) for each long, -gross/(2*k_eff) for each short.
    - Dollar-neutral: sum(w) = 0, sum(|w|) = gross_exposure.

    If fewer than 2*k assets have a valid feature at t, k_eff = n_valid // 2
    (reduced automatically). If n_valid < 2, no weights at that timestamp.

    Parameters
    ----------
    feature:
        Feature column to rank on. Default 'funding_zscore_30d' from B2.
    k:
        Number of long positions (and, symmetrically, of short positions).
        Scale with universe size: 2-5 for ~10 assets, 10-20 for 200+ assets.
        Change via constructor, CLI (`--k`), or YAML config.
    gross_exposure:
        sum(|w|) per rebalance. Default 1.0 (each side carries 0.5 of
        capital). Set to 2.0 for standard long/short equity sizing
        (each side carries 1.0).
    horizon_bars:
        Expected label horizon in bars. Must be <= rebalance_bars to
        avoid overlapping labels. Used for validation only.
    rebalance_bars:
        Expected rebalance frequency in bars. Must be >= horizon_bars.
        Used for validation only; the actual rebalance is driven by the
        `rebalance_freq` argument of `generate_weights`.

    Notes
    -----
    - No look-ahead: weights at t use only features at t (close of bar t),
      which are executed at t+1 open by the evaluator.
    - Funding itself is applied by the evaluator (B4) as a signed cash flow,
      not by this strategy. Shorts of high-funding assets should therefore
      receive funding; the evaluator enforces the sign convention.
    """

    def __init__(
        self,
        feature: str = "funding_zscore_30d",
        k: int = 2,
        gross_exposure: float = 1.0,
        horizon_bars: int = 8,
        rebalance_bars: int = 8,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        if gross_exposure <= 0:
            raise ValueError("gross_exposure must be > 0")
        if horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        if rebalance_bars < horizon_bars:
            raise ValueError(
                f"rebalance_bars ({rebalance_bars}) must be >= "
                f"horizon_bars ({horizon_bars}): labels would overlap "
                f"and uniqueness weights would be required"
            )
        self._feature = feature
        self._k = k
        self._gross = gross_exposure
        self._horizon = horizon_bars
        self._rebalance = rebalance_bars

    @property
    def name(self) -> str:
        return f"funding_longshort_topk_k{self._k}_h{self._horizon}_r{self._rebalance}"

    def generate_weights(self, df: pl.DataFrame, rebalance_freq: int) -> pl.DataFrame:
        """Return DataFrame[timestamp, asset_id, weight] at rebalance
        timestamps. Every `rebalance_freq` bars starting from the first."""
        if self._feature not in df.columns:
            raise ValueError(f"{self.name!r} requires column {self._feature!r}")
        if rebalance_freq < 1:
            raise ValueError("rebalance_freq must be >= 1")

        empty_schema = {
            "timestamp": pl.Datetime("us", "UTC"),
            "asset_id": pl.Utf8,
            "weight": pl.Float64,
        }

        all_ts = sorted(df["timestamp"].unique().to_list())
        if not all_ts:
            return pl.DataFrame(schema=empty_schema)

        rebalance_ts = all_ts[::rebalance_freq]
        sub = df.filter(pl.col("timestamp").is_in(rebalance_ts)).select(
            ["timestamp", "asset_id", self._feature]
        )

        rows: list[dict] = []
        for ts in rebalance_ts:
            snapshot = sub.filter(pl.col("timestamp") == ts).drop_nulls(self._feature)
            n_valid = snapshot.height
            if n_valid < 2:
                continue
            # Skip timestamps where all features are equal: no cross-sectional
            # signal (e.g. all zeros after std==0 fallback in B2, or all-tied
            # funding rates). Without this, the sort is ambiguous and
            # long/short assignment becomes arbitrary.
            if snapshot[self._feature].n_unique() < 2:
                continue
            k_eff = min(self._k, n_valid // 2)
            if k_eff < 1:
                continue
            # Deterministic tiebreaker: sort by (feature, asset_id) so that
            # equal feature values do not produce run-to-run differences.
            ordered = snapshot.sort([self._feature, "asset_id"])
            w = self._gross / (2 * k_eff)
            for row in ordered.head(k_eff).iter_rows(named=True):
                rows.append({"timestamp": ts, "asset_id": row["asset_id"], "weight": w})
            for row in ordered.tail(k_eff).iter_rows(named=True):
                rows.append(
                    {
                        "timestamp": ts,
                        "asset_id": row["asset_id"],
                        "weight": -w,
                    }
                )

        if not rows:
            return pl.DataFrame(schema=empty_schema)

        return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


@dataclass(frozen=True, slots=True)
class MomentumTrend:
    """Long-only conditional trend-following per asset.

    At each rebalance timestamp t:
      - Compute MA_N(close) per asset using only close up to t (no look-ahead).
      - Asset i gets weight 1 / n_active if close_i(t) > MA_N(close_i)(t).
      - Asset i gets weight 0 otherwise.
      - If no asset qualifies, all weights are 0 (portfolio in cash).

    Parameters
    ----------
    window : int
        Moving-average window in bars. Default 168 (7 days on 1h bars).
    price_col : str
        Column used for the moving average. Default "close".

    Notes
    -----
    This is NOT MomentumTopK with top_k = n_assets. MomentumTopK is
    market-neutral (long top-K / short bottom-K); MomentumTrend is
    long-only conditional. Mixing them would introduce an implicit
    directional bias that breaks the comparison against RandomTopK.
    """

    window: int = 168
    price_col: str = "close"
    _name: str = "momentum_trend"

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")

    @property
    def name(self) -> str:
        return f"momentum_trend_{self.window}"

    def generate_weights(self, df: pl.DataFrame, rebalance_freq: int) -> pl.DataFrame:
        if self.price_col not in df.columns:
            raise ValueError(f"price column {self.price_col!r} not in df")

        # MA per asset using only past/current bars. Polars rolling_mean
        # over an ordered window is causal by construction.
        df_ma = df.with_columns(
            pl.col(self.price_col)
            .rolling_mean(window_size=self.window, min_samples=self.window)
            .over("asset_id")
            .alias("_ma")
        )

        # Only rebalance where at least one asset has a valid MA.
        candidates = df_ma.filter(pl.col("_ma").is_not_null())
        if candidates.height == 0:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime("us", "UTC"),
                    "asset_id": pl.Utf8,
                    "weight": pl.Float64,
                }
            )

        reb = _rebalance_timestamps(candidates, rebalance_freq)

        def fn(group: pl.DataFrame) -> pl.DataFrame:
            valid = group.filter(pl.col("_ma").is_not_null())
            if valid.height == 0:
                return group.select("asset_id").with_columns(pl.lit(0.0).alias("weight"))

            signal = valid.select(
                "asset_id",
                (pl.col(self.price_col) > pl.col("_ma")).alias("_long"),
            )
            n_active = int(signal["_long"].sum())
            if n_active == 0:
                return signal.select("asset_id").with_columns(pl.lit(0.0).alias("weight"))

            weight = 1.0 / n_active
            return signal.select(
                "asset_id",
                pl.when(pl.col("_long"))
                .then(pl.lit(weight))
                .otherwise(pl.lit(0.0))
                .alias("weight"),
            )

        return _weights_at_rebalances(df_ma, reb, fn)
