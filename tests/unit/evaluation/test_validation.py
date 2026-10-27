"""Tests for canonical validation module (P0 commit 3)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from locus.evaluation.validation import (
    compute_dsr,
    compute_pbo,
    control_c1_random_weights,
    control_c2_permute_features,
    control_c3_shift_labels,
    paired_sharpe_diff,
    run_validation,
    stationary_bootstrap_sharpe_ci,
)

# --- DSR ---


def _iid_returns(mean: float, std: float, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, n)


def test_dsr_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        compute_dsr(np.array([0.01, 0.02]), n_trials=1, periods_per_year=252)


def test_dsr_rejects_bad_n_trials() -> None:
    with pytest.raises(ValueError, match="n_trials"):
        compute_dsr(_iid_returns(0.001, 0.01, 200), n_trials=0, periods_per_year=252)


def test_dsr_rejects_zero_variance() -> None:
    with pytest.raises(ValueError, match="zero standard deviation"):
        compute_dsr(np.zeros(100), n_trials=1, periods_per_year=252)


def test_dsr_high_sharpe_low_trials_is_significant() -> None:
    r = _iid_returns(0.003, 0.01, 500)
    res = compute_dsr(r, n_trials=1, periods_per_year=252)
    assert res.dsr > 0.99
    assert res.is_significant


def test_dsr_high_sharpe_many_trials_is_less_significant() -> None:
    r = _iid_returns(0.003, 0.01, 500)
    low = compute_dsr(r, n_trials=1, periods_per_year=252)
    high = compute_dsr(r, n_trials=1000, periods_per_year=252)
    assert high.dsr < low.dsr
    assert high.sr_zero > low.sr_zero


def test_dsr_zero_signal_not_significant() -> None:
    r = _iid_returns(0.0, 0.01, 500)
    res = compute_dsr(r, n_trials=10, periods_per_year=252)
    assert not res.is_significant


def test_dsr_with_provided_variance() -> None:
    r = _iid_returns(0.003, 0.01, 500)
    res = compute_dsr(
        r,
        n_trials=50,
        periods_per_year=252,
        sharpe_variance_across_trials=0.02,
    )
    assert 0 <= res.dsr <= 1


# --- PBO ---


def _returns_matrix(
    n_obs: int,
    n_strats: int,
    *,
    signal_strength: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.01, size=(n_obs, n_strats))
    if signal_strength > 0:
        base[:, 0] += signal_strength
    return base


def test_pbo_rejects_bad_s() -> None:
    returns_mat = _returns_matrix(100, 5)
    with pytest.raises(ValueError, match="s_blocks"):
        compute_pbo(returns_mat, s_blocks=3)
    with pytest.raises(ValueError, match="s_blocks"):
        compute_pbo(returns_mat, s_blocks=2)


def test_pbo_rejects_single_column() -> None:
    with pytest.raises(ValueError, match=">= 2 strategies"):
        compute_pbo(np.random.randn(100, 1))


def test_pbo_high_for_random_strategies() -> None:
    returns_mat = _returns_matrix(1000, 10, signal_strength=0.0)
    res = compute_pbo(returns_mat, s_blocks=8)
    assert 0.2 <= res.pbo <= 0.9


def test_pbo_low_for_strategy_with_clear_edge() -> None:
    returns_mat = _returns_matrix(2000, 5, signal_strength=0.01)
    res = compute_pbo(returns_mat, s_blocks=8)
    assert res.pbo < 0.7


def test_pbo_output_shape() -> None:
    returns_mat = _returns_matrix(1000, 5)
    res = compute_pbo(returns_mat, s_blocks=8)
    assert res.s_blocks == 8
    assert res.n_combinations == 70
    assert res.logits.shape == (70,)


# --- Stationary bootstrap ---


def test_bootstrap_ci_contains_point_estimate() -> None:
    r = _iid_returns(0.001, 0.01, 500)
    lo, hi = stationary_bootstrap_sharpe_ci(r, 252, n_boot=500)
    observed = (r.mean() / r.std(ddof=1)) * np.sqrt(252)
    assert lo < observed < hi


def test_bootstrap_ci_narrows_with_more_data() -> None:
    """CI width scales with 1/sqrt(n): 4x more data → ~2x narrower CI."""
    r_small = _iid_returns(0.001, 0.01, 500, seed=1)
    r_large = _iid_returns(0.001, 0.01, 2000, seed=1)
    lo1, hi1 = stationary_bootstrap_sharpe_ci(
        r_small, 252, n_boot=1000, rng=np.random.default_rng(1)
    )
    lo2, hi2 = stationary_bootstrap_sharpe_ci(
        r_large, 252, n_boot=1000, rng=np.random.default_rng(1)
    )
    width_small = hi1 - lo1
    width_large = hi2 - lo2
    assert width_large < width_small


def test_bootstrap_ci_contains_observed_sharpe() -> None:
    """CI contains the observed Sharpe for a range of return volatilities."""
    for std in (0.005, 0.01, 0.02):
        r = _iid_returns(0.001, std, 500, seed=1)
        lo, hi = stationary_bootstrap_sharpe_ci(r, 252, n_boot=500, rng=np.random.default_rng(1))
        observed = (r.mean() / r.std(ddof=1)) * np.sqrt(252)
        assert lo < observed < hi, f"std={std}: observed {observed:.3f} not in [{lo:.3f}, {hi:.3f}]"


# --- Paired diff ---


def test_paired_diff_same_series_is_zero() -> None:
    r = _iid_returns(0.001, 0.01, 500)
    diff, p, _ = paired_sharpe_diff(r, r, 252, n_boot=500)
    assert abs(diff) < 1e-10
    assert p == 1.0


def test_paired_diff_better_a_than_b() -> None:
    rng = np.random.default_rng(42)
    b = rng.normal(0.0, 0.01, 500)
    a = rng.normal(0.005, 0.01, 500)
    diff, p, _ = paired_sharpe_diff(a, b, 252, n_boot=500)
    assert diff > 0
    assert p < 0.05


def test_paired_diff_rejects_shape_mismatch() -> None:
    a = np.zeros(100)
    b = np.zeros(99)
    with pytest.raises(ValueError, match="same shape"):
        paired_sharpe_diff(a, b, 252)


# --- Controls ---


def _weights_df() -> pl.DataFrame:
    from datetime import UTC, datetime, timedelta

    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for t in range(3):
        ts = base + timedelta(days=t)
        for asset in ["A", "B", "C", "D"]:
            rows.append(
                {
                    "timestamp": ts,
                    "asset_id": asset,
                    "weight": 0.25 if asset in ["A", "B"] else -0.25,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def test_c1_preserves_per_timestamp_gross() -> None:
    weights = _weights_df()
    rand = control_c1_random_weights(weights)
    for ts in weights["timestamp"].unique().to_list():
        g_orig = weights.filter(pl.col("timestamp") == ts)["weight"].abs().sum()
        g_rand = rand.filter(pl.col("timestamp") == ts)["weight"].abs().sum()
        assert abs(g_orig - g_rand) < 1e-9


def test_c1_preserves_per_timestamp_net() -> None:
    weights = _weights_df()
    rand = control_c1_random_weights(weights)
    for ts in weights["timestamp"].unique().to_list():
        n_orig = weights.filter(pl.col("timestamp") == ts)["weight"].sum()
        n_rand = rand.filter(pl.col("timestamp") == ts)["weight"].sum()
        assert abs(n_orig - n_rand) < 1e-9


def test_c2_permutes_values() -> None:
    df = pl.DataFrame({"asset_id": ["A"] * 100, "f": list(range(100))})
    out = control_c2_permute_features(df, ["f"], seed=1)
    assert sorted(out["f"].to_list()) == sorted(df["f"].to_list())
    assert out["f"].to_list() != df["f"].to_list()


def test_c2_rejects_missing_column() -> None:
    df = pl.DataFrame({"f": [1, 2, 3]})
    with pytest.raises(ValueError, match="not in DataFrame"):
        control_c2_permute_features(df, ["missing"])


def test_c3_shifts_labels() -> None:
    s = pl.Series("y", [1, 2, 3, 4, 5])
    shifted = control_c3_shift_labels(s, shift=1)
    assert shifted.to_list() == [None, 1, 2, 3, 4]


# --- run_validation ---


def test_run_validation_positive_case() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.005, 0.01, 500)
    baseline = rng.normal(0.0, 0.01, 500)
    returns_matrix = np.column_stack([returns, rng.normal(0, 0.01, 500), rng.normal(0, 0.01, 500)])
    report = run_validation(
        returns,
        returns_matrix,
        baseline,
        n_trials=3,
        periods_per_year=252,
        n_boot=500,
    )
    assert report.verdict == "positive"
    assert report.dsr.is_significant
    assert report.sharpe_ci_low > 0


def test_run_validation_negative_case() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, 500)
    baseline = rng.normal(0.0, 0.01, 500)
    returns_matrix = np.column_stack([returns, rng.normal(0, 0.01, 500)])
    report = run_validation(
        returns,
        returns_matrix,
        baseline,
        n_trials=2,
        periods_per_year=252,
        n_boot=500,
    )
    assert report.verdict in ("negative", "inconclusive")


def test_run_validation_rationale_non_empty() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, 500)
    baseline = rng.normal(0.0, 0.01, 500)
    returns_matrix = np.column_stack([returns, baseline])
    report = run_validation(
        returns,
        returns_matrix,
        baseline,
        n_trials=2,
        periods_per_year=252,
        n_boot=200,
    )
    assert report.rationale
    assert report.verdict in ("positive", "negative", "inconclusive")
