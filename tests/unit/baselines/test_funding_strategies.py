"""
test_funding_strategy.py

Tests for FundingLongShortTopK (B5) and the annualization_days
parameter (B5 SHOULD).

Author: Marco Pérez Padilla
Date:   25-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.backtest.config import BacktestConfig
from locus.backtest.costs import (
    FeeModel,
    MarketImpactModel,
    SlippageModel,
    SpreadModel,
    TransactionCostModel,
)
from locus.baselines.strategies import FundingLongShortTopK
from locus.data.providers.base import BAR_SCHEMA

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _bars_and_features(
    feature_values: dict[str, list[float]],
    n_bars: int | None = None,
) -> pl.DataFrame:
    """Build a bars DataFrame with a `funding_zscore_30d` column.

    Each asset's feature value repeats for `step` bars so that
    rebalance_freq=step aligns with distinct feature levels.
    """
    if n_bars is None:
        n_bars = len(next(iter(feature_values.values())))
    base = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for asset, values in feature_values.items():
        if len(values) != n_bars:
            raise ValueError("all feature series must be length n_bars")
        for i, v in enumerate(values):
            rows.append(
                {
                    "asset_id": asset,
                    "venue": "test",
                    "symbol": f"{asset}USDT",
                    "timeframe": "1h",
                    "timestamp": base + timedelta(hours=i),
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                    "volume": 1.0,
                    "funding_zscore_30d": v,
                }
            )
    return pl.DataFrame(
        rows,
        schema_overrides=BAR_SCHEMA
        | {
            "funding_zscore_30d": pl.Float64,
        },
    )


def _sorted_weights(w: pl.DataFrame) -> pl.DataFrame:
    return w.sort(["timestamp", "asset_id"])


# ----------------------------------------------------------------------
# FundingLongShortTopK
# ----------------------------------------------------------------------


def test_longshort_long_low_funding_short_high_funding() -> None:
    df = _bars_and_features(
        {
            "A": [3.0],
            "B": [1.0],
            "C": [-1.0],
            "D": [-3.0],
        }
    )
    strategy = FundingLongShortTopK(k=1, gross_exposure=1.0)
    weights = strategy.generate_weights(df, rebalance_freq=1).sort("asset_id")
    by_asset = {r["asset_id"]: r["weight"] for r in weights.iter_rows(named=True)}
    # Lowest funding (D, -3.0) is the long (+0.5); highest (A, 3.0) is the short (-0.5).
    assert by_asset["D"] == pytest.approx(+0.5)
    assert by_asset["A"] == pytest.approx(-0.5)
    assert "B" not in by_asset and "C" not in by_asset


def test_longshort_dollar_neutral_and_gross_exposure() -> None:
    df = _bars_and_features(
        {
            "A": [1.0],
            "B": [2.0],
            "C": [-1.0],
            "D": [-2.0],
        }
    )
    strategy = FundingLongShortTopK(k=2, gross_exposure=1.0)
    weights = strategy.generate_weights(df, rebalance_freq=1)
    total = weights["weight"].sum()
    gross = weights["weight"].abs().sum()
    assert total == pytest.approx(0.0, abs=1e-12)
    assert gross == pytest.approx(1.0, abs=1e-12)


def test_longshort_gross_exposure_scales() -> None:
    df = _bars_and_features({"A": [1.0], "B": [2.0], "C": [-1.0], "D": [-2.0]})
    strategy = FundingLongShortTopK(k=2, gross_exposure=2.0)
    weights = strategy.generate_weights(df, rebalance_freq=1)
    assert weights["weight"].abs().sum() == pytest.approx(2.0, abs=1e-12)
    assert weights["weight"].sum() == pytest.approx(0.0, abs=1e-12)


def test_longshort_k_too_large_reduces_to_floor_n_over_2() -> None:
    df = _bars_and_features({"A": [1.0], "B": [2.0], "C": [3.0], "D": [4.0]})
    strategy = FundingLongShortTopK(k=5, gross_exposure=1.0)
    weights = strategy.generate_weights(df, rebalance_freq=1)
    # k_eff = min(5, 4 // 2) = 2 → 4 positions.
    assert weights.height == 4
    assert weights.filter(pl.col("weight") > 0).height == 2
    assert weights.filter(pl.col("weight") < 0).height == 2


def test_longshort_fewer_than_two_valid_assets_no_weights() -> None:
    df = _bars_and_features({"A": [1.0], "B": [None]})
    strategy = FundingLongShortTopK(k=1)
    weights = strategy.generate_weights(df, rebalance_freq=1)
    assert weights.height == 0


def test_longshort_warmup_nulls_dropped() -> None:
    df = _bars_and_features(
        {
            "A": [None, None, 1.0],
            "B": [None, None, 2.0],
            "C": [None, None, 3.0],
            "D": [None, None, 4.0],
        }
    )
    strategy = FundingLongShortTopK(k=1)
    weights = strategy.generate_weights(df, rebalance_freq=1)
    # Only the third timestamp has non-null features for all four.
    assert weights["timestamp"].n_unique() == 1


def test_longshort_no_leakage_under_truncation() -> None:
    feature_values = {
        "A": [float(i) for i in range(1, 41)],
        "B": [float(i) for i in range(2, 42)],
        "C": [float(i) for i in range(3, 43)],
        "D": [float(i) for i in range(4, 44)],
    }
    df = _bars_and_features(feature_values)
    strategy = FundingLongShortTopK(k=1, gross_exposure=1.0)

    full = strategy.generate_weights(df, rebalance_freq=4)
    cutoff_ts = sorted(df["timestamp"].unique().to_list())[20]
    truncated = strategy.generate_weights(
        df.filter(pl.col("timestamp") <= cutoff_ts), rebalance_freq=4
    )

    for ts in truncated["timestamp"].unique().to_list():
        w_full = _sorted_weights(full.filter(pl.col("timestamp") == ts))
        w_trunc = _sorted_weights(truncated.filter(pl.col("timestamp") == ts))
        assert w_full.equals(w_trunc), f"leakage at {ts}"


def test_longshort_horizon_rebalance_validation() -> None:
    with pytest.raises(ValueError, match="rebalance_bars"):
        FundingLongShortTopK(horizon_bars=8, rebalance_bars=1)
    FundingLongShortTopK(horizon_bars=8, rebalance_bars=8)
    FundingLongShortTopK(horizon_bars=8, rebalance_bars=24)


def test_longshort_invalid_construction() -> None:
    with pytest.raises(ValueError):
        FundingLongShortTopK(k=0)
    with pytest.raises(ValueError):
        FundingLongShortTopK(gross_exposure=0.0)
    with pytest.raises(ValueError):
        FundingLongShortTopK(horizon_bars=0)


# ----------------------------------------------------------------------
# annualization_days (B5 SHOULD)
# ----------------------------------------------------------------------


def _config(**overrides) -> BacktestConfig:
    base = {
        "strategy_name": "test",
        "capital": 1_000.0,
        "currency": "USDT",
        "cost_model": TransactionCostModel(
            fee=FeeModel(0.0),
            spread=SpreadModel(0.0),
            slippage=SlippageModel(0.0),
            impact=MarketImpactModel(0.0),
        ),
        "min_order_notional": 0.0,
        "default_lot_size": 1e-8,
        "lot_sizes": {},
    }
    base.update(overrides)
    return BacktestConfig(**base)


def test_periods_per_year_default_365() -> None:
    cfg = _config(bars_per_day=24, rebalance_freq_bars=8, annualization_days=365)
    assert cfg.periods_per_year == 1095


def test_periods_per_year_custom_252() -> None:
    cfg = _config(bars_per_day=24, rebalance_freq_bars=8, annualization_days=252)
    assert cfg.periods_per_year == 756


def test_periods_per_year_invalid() -> None:
    with pytest.raises(ValueError, match="annualization_days"):
        _config(annualization_days=0)


def test_longshort_skips_uniform_feature_timestamp() -> None:
    """If all assets have the same feature value at a timestamp, there is
    no cross-sectional signal: emit no weights for that timestamp."""
    df = _bars_and_features({"A": [0.0], "B": [0.0], "C": [0.0], "D": [0.0]})
    strategy = FundingLongShortTopK(k=2)
    weights = strategy.generate_weights(df, rebalance_freq=1)
    assert weights.height == 0


def test_longshort_uniform_first_skips_but_second_trades() -> None:
    """Mixed case: uniform at t0 (skipped), non-uniform at t1 (traded)."""
    df = _bars_and_features(
        {
            "A": [0.0, 1.0],
            "B": [0.0, 2.0],
            "C": [0.0, 3.0],
            "D": [0.0, 4.0],
        }
    )
    strategy = FundingLongShortTopK(k=1)
    weights = strategy.generate_weights(df, rebalance_freq=1)
    assert weights["timestamp"].n_unique() == 1  # only t1 has weights


def test_longshort_tiebreaker_is_deterministic() -> None:
    """Tied feature values are broken by asset_id, so identical inputs
    produce identical long/short assignment across runs."""
    df = _bars_and_features(
        {
            "A": [1.0],
            "B": [1.0],
            "C": [-1.0],
            "D": [-1.0],
        }
    )
    strategy = FundingLongShortTopK(k=1)
    w1 = strategy.generate_weights(df, rebalance_freq=1).sort("asset_id")
    w2 = strategy.generate_weights(df, rebalance_freq=1).sort("asset_id")
    assert w1.equals(w2)
    by_asset = {r["asset_id"]: r["weight"] for r in w1.iter_rows(named=True)}
    assert by_asset["C"] > 0
    assert by_asset["B"] < 0
