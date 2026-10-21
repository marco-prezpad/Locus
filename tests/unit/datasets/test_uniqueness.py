"""
test_uniqueness.py

Uniqueness weights: 1/concurrency per asset, sum = n_eff, uniform
spacing gives close to 1/horizon.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.data.providers.base import BAR_SCHEMA
from locus.datasets import compute_uniqueness_weights, estimate_n_eff_simple


def _df(n: int, asset: str = "A") -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
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
            }
            for i in range(n)
        ],
        schema=BAR_SCHEMA,
    )


def test_interior_weights_equal_one_over_horizon() -> None:
    df = _df(200)
    weighted = compute_uniqueness_weights(df, horizon_bars=24)
    weights = weighted["uniqueness_weight"].to_list()
    # Interior rows (not near boundaries) get 1/24
    interior = weights[50:150]
    assert all(w == pytest.approx(1.0 / 24) for w in interior)


def test_per_asset_isolation() -> None:
    df = pl.concat([_df(100, "A"), _df(100, "B")])
    weighted = compute_uniqueness_weights(df, horizon_bars=24)
    a = weighted.filter(pl.col("asset_id") == "A")["uniqueness_weight"].to_list()
    b = weighted.filter(pl.col("asset_id") == "B")["uniqueness_weight"].to_list()
    # Same pattern for both, no cross-contamination
    assert a == b


def test_n_eff_simple_approximation() -> None:
    assert estimate_n_eff_simple(2400, 24) == pytest.approx(100.0)


def test_first_rows_have_higher_weight_than_interior() -> None:
    """The first h-1 rows have fewer past labels covering them, so
    their weight is higher than the interior 1/h. This is the only
    boundary effect under the half-open [t, t+h) convention: the last
    row has the same density as the interior (there are always h past
    labels covering it)."""
    df = _df(200)
    weighted = compute_uniqueness_weights(df, horizon_bars=24)
    weights = weighted["uniqueness_weight"].to_list()
    assert weights[0] > weights[100]
    assert weights[1] > weights[100]
    assert weights[-1] == pytest.approx(weights[100])


def test_first_h_weights_follow_harmonic_series() -> None:
    """Rows 0..h-1 have weights 1/1, 1/2, ..., 1/h, then the interior
    settles at 1/h."""
    df = _df(200)
    weighted = compute_uniqueness_weights(df, horizon_bars=24)
    weights = weighted["uniqueness_weight"].to_list()
    for i in range(24):
        assert weights[i] == pytest.approx(1.0 / (i + 1))
    # From row 23 onward, always 1/24
    for i in range(23, 200):
        assert weights[i] == pytest.approx(1.0 / 24)


def test_weights_sum_close_to_n_over_h() -> None:
    """For a long series, sum(weights) ≈ n/h plus a small boundary
    correction (the first h rows are slightly heavier than 1/h)."""
    df = _df(2400)
    weighted = compute_uniqueness_weights(df, horizon_bars=24)
    total = float(weighted["uniqueness_weight"].sum())
    # Interior contribution: (n - h) / h ≈ 99. Boundary contribution
    # (first 24 rows): sum_{i=1..24} 1/i ≈ 3.78. Total ≈ 102.8.
    expected = (2400 - 24) / 24 + sum(1.0 / i for i in range(1, 25))
    assert total == pytest.approx(expected, rel=0.01)


def test_rejects_zero_horizon() -> None:
    with pytest.raises(ValueError):
        compute_uniqueness_weights(_df(10), horizon_bars=0)
    with pytest.raises(ValueError):
        estimate_n_eff_simple(100, 0)
