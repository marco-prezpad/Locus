"""
validation.py

Canonical statistical validation for experiment verdicts (plan v3.1 P0).

Implements:
- DSR (Deflated Sharpe Ratio) per Bailey & López de Prado (2014)
- PBO (Probability of Backtest Overfitting) per Bailey et al. (2015), CSCV
- Stationary block bootstrap CI for Sharpe (Politis & Romano 1994)
- Paired Sharpe difference with stationary bootstrap
- Control utilities C1 (random weights), C2 (permuted features),
  C3 (shifted labels)

Verdict policy (plan §7.4): a strategy cannot be called positive without
DSR and PBO against the accumulated n_trials. This module is the only
source of those numbers.

The legacy `block_bootstrap_sharpe_ci` in `significance.py` is retained
for backward compatibility but is NOT used for verdicts (plan §6.1:
"Signification: Aproximada; debe sustituirse por DSR canónico y PBO").
Verdicts use this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import norm
from scipy.stats import skew as scipy_skew

_EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# DSR (Deflated Sharpe Ratio)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DSRResult:
    """Outcome of the Deflated Sharpe Ratio test.

    `dsr` is a probability in [0, 1] that the observed Sharpe exceeds the
    threshold SR0 (the expected maximum under the null given n_trials).
    Convention: dsr > 0.95 means the observed Sharpe is significant at 5%.
    """

    dsr: float
    sr_observed: float
    sr_zero: float
    n_trials: int
    n_obs: int
    is_significant: bool


def compute_dsr(
    returns: np.ndarray,
    n_trials: int,
    periods_per_year: int,
    *,
    sharpe_variance_across_trials: float | None = None,
    significance_threshold: float = 0.95,
) -> DSRResult:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Parameters
    ----------
    returns : np.ndarray
        Per-period returns of the strategy. Must be finite, length >= 3.
    n_trials : int
        Total number of configurations tested, from the trial registry.
    periods_per_year : int
        Annualization factor (252 for daily, 8760 for hourly, etc.).
    sharpe_variance_across_trials : float | None
        Variance of Sharpe ratios across the n_trials. If None, uses the
        conservative estimate V[SR] = 1 / (T - 1) under the null of unit
        variance. Providing the actual variance is more accurate.
    significance_threshold : float
        DSR value above which the result is flagged significant
        (default 0.95).

    Returns
    -------
    DSRResult with dsr in [0, 1].

    References
    ----------
    Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
    Journal of Portfolio Management, 40(5), 94-107.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n_obs = arr.size
    if n_obs < 3:
        raise ValueError(f"need at least 3 finite returns, got {n_obs}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0:
        raise ValueError("returns have zero standard deviation")
    sr_per_period = mu / sd

    g3 = float(scipy_skew(arr, bias=False))
    g4 = float(scipy_kurtosis(arr, fisher=False, bias=False))

    if sharpe_variance_across_trials is None:
        v_sr = 1.0 / (n_obs - 1)
    else:
        v_sr = float(sharpe_variance_across_trials)
        if v_sr <= 0:
            raise ValueError("sharpe_variance_across_trials must be > 0")

    e = np.e
    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * e))
    sr_zero = np.sqrt(v_sr) * ((1 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)

    denom = 1 - g3 * sr_per_period + (g4 - 1) / 4 * sr_per_period**2
    if denom <= 0:
        raise ValueError(
            f"denominator non-positive ({denom}); returns are degenerate "
            f"(skewness={g3}, kurtosis={g4})"
        )
    z_stat = (sr_per_period - sr_zero) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    dsr = float(norm.cdf(z_stat))

    return DSRResult(
        dsr=dsr,
        sr_observed=sr_per_period * np.sqrt(periods_per_year),
        sr_zero=float(sr_zero) * np.sqrt(periods_per_year),
        n_trials=n_trials,
        n_obs=n_obs,
        is_significant=dsr >= significance_threshold,
    )


# ---------------------------------------------------------------------------
# PBO (Probability of Backtest Overfitting) — CSCV
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PBOResult:
    pbo: float
    n_combinations: int
    logits: np.ndarray
    s_blocks: int


def compute_pbo(
    returns_matrix: np.ndarray,
    s_blocks: int = 8,
) -> PBOResult:
    """Probability of Backtest Overfitting via CSCV (Bailey et al. 2015).

    Parameters
    ----------
    returns_matrix : np.ndarray
        Shape (n_obs, n_strats). Each column is one strategy's per-period
        returns.
    s_blocks : int
        Number of blocks to split n_obs into. Must be even and >= 4.
        Paper recommends 8 or 10.

    Returns
    -------
    PBOResult with pbo in [0, 1]. pbo > 0.5 suggests the best in-sample
    strategy is likely below median out-of-sample.

    References
    ----------
    Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2015).
    "The Probability of Backtest Overfitting." Journal of Computational
    Finance, 20(4), 39-69.
    """
    from itertools import combinations

    matrix = np.asarray(returns_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"returns_matrix must be 2D, got ndim={matrix.ndim}")
    n_obs, n_strats = matrix.shape
    if n_strats < 2:
        raise ValueError(f"need >= 2 strategies (columns), got {n_strats}")
    if s_blocks < 4 or s_blocks % 2 != 0:
        raise ValueError(f"s_blocks must be even and >= 4, got {s_blocks}")
    if n_obs < s_blocks * 2:
        raise ValueError(f"need at least {s_blocks * 2} observations, got {n_obs}")

    block_size = n_obs // s_blocks
    blocks = [matrix[i * block_size : (i + 1) * block_size] for i in range(s_blocks)]
    half = s_blocks // 2

    def _sharpe(cols: np.ndarray) -> np.ndarray:
        mu = cols.mean(axis=0)
        sd = cols.std(axis=0, ddof=1)
        sd = np.where(sd == 0, 1.0, sd)
        return mu / sd

    logits: list[float] = []
    for is_idx in combinations(range(s_blocks), half):
        oos_idx = tuple(i for i in range(s_blocks) if i not in is_idx)
        is_mat = np.concatenate([blocks[i] for i in is_idx], axis=0)
        oos_mat = np.concatenate([blocks[i] for i in oos_idx], axis=0)

        sr_is = _sharpe(is_mat)
        sr_oos = _sharpe(oos_mat)

        best_is = int(np.argmax(sr_is))
        oos_order = np.argsort(sr_oos)
        rank = int(np.where(oos_order == best_is)[0][0]) + 1
        omega = rank / (n_strats + 1)
        logit = float(np.log(omega / (1 - omega)))
        logits.append(logit)

    logits_arr = np.array(logits)
    pbo = float((logits_arr < 0).mean())
    return PBOResult(
        pbo=pbo,
        n_combinations=len(logits),
        logits=logits_arr,
        s_blocks=s_blocks,
    )


# ---------------------------------------------------------------------------
# Stationary bootstrap (Politis & Romano 1994)
# ---------------------------------------------------------------------------


def _stationary_bootstrap_indices(
    n: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stationary bootstrap (Politis & Romano 1994) with geometric blocks.

    Returns an index array of length n resampled from range(n).
    """
    p = 1.0 / block_size
    idx = np.empty(n, dtype=np.int64)
    i = int(rng.integers(0, n))
    for t in range(n):
        if rng.random() < p:
            i = int(rng.integers(0, n))
        idx[t] = i
        i = (i + 1) % n
    return idx


def stationary_bootstrap_sharpe_ci(
    returns: np.ndarray,
    periods_per_year: int,
    *,
    block_size: int = 7,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Stationary block bootstrap CI for annualized Sharpe.

    Returns (lower, upper) at confidence level 1 - alpha.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 10:
        raise ValueError(f"need at least 10 returns, got {n}")
    rng = rng or np.random.default_rng(42)

    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(n, block_size, rng)
        sample = arr[idx]
        sd = sample.std(ddof=1)
        sharpes[b] = 0.0 if sd == 0 else sample.mean() / sd
    sharpes *= np.sqrt(periods_per_year)
    lower = float(np.quantile(sharpes, alpha / 2))
    upper = float(np.quantile(sharpes, 1 - alpha / 2))
    return lower, upper


def paired_sharpe_diff(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    periods_per_year: int,
    *,
    block_size: int = 7,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI and one-sided p-value for Sharpe(A) - Sharpe(B).

    Uses paired stationary block bootstrap (same indices for A and B).

    Returns (mean_diff, p_value, ci_low) where p_value is the fraction
    of bootstrap samples with diff <= 0 (H1: A > B).
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"return arrays must have same shape, got {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    if n < 10:
        raise ValueError(f"need at least 10 paired returns, got {n}")
    rng = rng or np.random.default_rng(42)

    def _sharpe(x: np.ndarray) -> float:
        sd = x.std(ddof=1)
        return 0.0 if sd == 0 else x.mean() / sd

    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = _stationary_bootstrap_indices(n, block_size, rng)
        diffs[k] = _sharpe(a[idx]) - _sharpe(b[idx])
    diffs *= np.sqrt(periods_per_year)

    observed = (_sharpe(a) - _sharpe(b)) * np.sqrt(periods_per_year)
    p_value = float((diffs <= 0).mean())
    ci_low = float(np.quantile(diffs, 0.05))
    return observed, p_value, ci_low


# ---------------------------------------------------------------------------
# Controls C1, C2, C3 (plan §7.3)
# ---------------------------------------------------------------------------


def control_c1_random_weights(
    weights: pl.DataFrame,
    *,
    seed: int = 42,
) -> pl.DataFrame:
    """C1: random weights matching gross, net exposure and turnover.

    For each timestamp, shuffle the weight values across assets. Preserves
    gross exposure (sum of |w|) and net exposure (sum of w) per timestamp,
    but destroys the cross-sectional selection.
    """
    rng = np.random.default_rng(seed)
    out_rows: list[dict] = []
    for (ts,), group in weights.group_by(["timestamp"], maintain_order=True):
        w = group["weight"].to_numpy()
        perm = rng.permutation(len(w))
        for i, asset in enumerate(group["asset_id"].to_list()):
            out_rows.append({"timestamp": ts, "asset_id": asset, "weight": w[perm[i]]})
    return pl.DataFrame(
        out_rows,
        schema={
            "timestamp": weights.schema["timestamp"],
            "asset_id": pl.Utf8,
            "weight": pl.Float64,
        },
    )


def control_c2_permute_features(
    df: pl.DataFrame,
    feature_cols: list[str],
    *,
    seed: int = 42,
) -> pl.DataFrame:
    """C2: permute feature values across time (destroys temporal structure).

    Each feature column is independently shuffled along the time axis.
    Asset identity is preserved.
    """
    rng = np.random.default_rng(seed)
    out = df.clone()
    for col in feature_cols:
        if col not in out.columns:
            raise ValueError(f"feature column {col!r} not in DataFrame")
        values = out[col].to_numpy()
        out = out.with_columns(pl.Series(col, rng.permutation(values)))
    return out


def control_c3_shift_labels(
    labels: pl.Series,
    shift: int = 1,
) -> pl.Series:
    """C3: shift labels by one bar (should destroy predictive signal).

    Positive `shift` moves labels forward in time. A correctly constructed
    label series, when shifted, should be unpredictable from the original
    features.
    """
    return labels.shift(shift)


# ---------------------------------------------------------------------------
# Bundle: run all validation checks and emit a verdict
# ---------------------------------------------------------------------------


Verdict = Literal["positive", "negative", "inconclusive"]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Bundle of all validation checks for a single strategy."""

    sharpe: float
    sharpe_ci_low: float
    sharpe_ci_high: float
    dsr: DSRResult
    pbo: PBOResult
    paired_p_vs_baseline: float
    paired_observed_diff: float
    n_trials: int
    verdict: Verdict
    rationale: str


def run_validation(
    returns: np.ndarray,
    returns_matrix: np.ndarray,
    baseline_returns: np.ndarray,
    n_trials: int,
    periods_per_year: int,
    *,
    s_blocks: int = 8,
    block_size: int = 7,
    n_boot: int = 2000,
    dsr_threshold: float = 0.95,
    pbo_threshold: float = 0.50,
    seed: int = 42,
) -> ValidationReport:
    """Full validation report: Sharpe + CI + DSR + PBO + paired vs baseline.

    Verdict policy (aligned with plan §7.4, criteria E1, E2):
    - positive: dsr.is_significant AND pbo.pbo < pbo_threshold AND paired
                diff > 0 with p < 0.05 AND sharpe_ci_low > 0
    - negative: paired_observed_diff <= 0 OR sharpe_ci_low <= 0
    - inconclusive: otherwise
    """
    rng = np.random.default_rng(seed)
    finite = returns[np.isfinite(returns)]
    sd = finite.std(ddof=1)
    if sd == 0:
        raise ValueError("returns have zero standard deviation")
    observed_sharpe = (finite.mean() / sd) * np.sqrt(periods_per_year)

    ci_low, ci_high = stationary_bootstrap_sharpe_ci(
        returns,
        periods_per_year,
        block_size=block_size,
        n_boot=n_boot,
        rng=rng,
    )
    dsr = compute_dsr(
        returns,
        n_trials,
        periods_per_year,
        significance_threshold=dsr_threshold,
    )
    pbo = compute_pbo(returns_matrix, s_blocks=s_blocks)
    diff, p_val, _ = paired_sharpe_diff(
        returns,
        baseline_returns,
        periods_per_year,
        block_size=block_size,
        n_boot=n_boot,
        rng=rng,
    )

    if dsr.is_significant and pbo.pbo < pbo_threshold and diff > 0 and p_val < 0.05 and ci_low > 0:
        verdict: Verdict = "positive"
        rationale = (
            f"DSR={dsr.dsr:.3f} (sig), PBO={pbo.pbo:.3f} (<{pbo_threshold}), "
            f"paired diff={diff:+.3f} (p={p_val:.3f}), CI_low={ci_low:+.3f}"
        )
    elif diff <= 0 or ci_low <= 0:
        verdict = "negative"
        rationale = f"paired diff={diff:+.3f} (p={p_val:.3f}) or CI_low={ci_low:+.3f} non-positive"
    else:
        verdict = "inconclusive"
        rationale = (
            f"partial pass: DSR={dsr.dsr:.3f}, PBO={pbo.pbo:.3f}, "
            f"diff={diff:+.3f} (p={p_val:.3f}), CI_low={ci_low:+.3f}"
        )

    return ValidationReport(
        sharpe=observed_sharpe,
        sharpe_ci_low=ci_low,
        sharpe_ci_high=ci_high,
        dsr=dsr,
        pbo=pbo,
        paired_p_vs_baseline=p_val,
        paired_observed_diff=diff,
        n_trials=n_trials,
        verdict=verdict,
        rationale=rationale,
    )
