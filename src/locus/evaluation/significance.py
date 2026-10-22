"""
significance.py

Statistical tests for strategy Sharpe significance:

- Deflated Sharpe Ratio (DSR): Bailey & López de Prado (2014). Adjusts
  the observed Sharpe for the number of trials, skewness, kurtosis,
  and non-normality. Returns the probability that the true Sharpe > 0.
- Probability of Backtest Overfitting (PBO): Bailey, Borwein, López de
  Prado & Zhu (2017), via CSCV (Combinatorially Symmetric Cross-
  Validation). Returns the fraction of times the best in-sample
  strategy underperforms the median out-of-sample.
- Minimum Backtest Length (MinBTL): Bailey et al. Minimum number of
  years of data needed for a Sharpe to be distinguishable from the
  expected max under N trials.
- Block bootstrap Sharpe CI: moving-block bootstrap to produce a
  confidence interval that preserves autocorrelation.

All functions accept per-period returns (not annualized). They return
annualization-free quantities except MinBTL, which is in years.

References:
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017).
  The Probability of Backtest Overfitting. Journal of Computational
  Finance, 20(4), 39-69.
- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J.
  (2014). Pseudo-Mathematics and Financial Charlatanism.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import kurtosis, norm, skew

_EULER_GAMMA = 0.5772156649015329


def _expected_max_sharpe_under_null(n_trials: int) -> float:
    """E[max SR] per unit of standard deviation, under the null hypothesis
    of N independent trials with zero Sharpe.

    Uses the Bailey-López de Prado approximation:
    E[max Z_N] ≈ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    n = float(n_trials)
    return float(
        (1 - _EULER_GAMMA) * norm.ppf(1 - 1 / n) + _EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e))
    )


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
) -> float:
    """Deflated Sharpe Ratio: probability that the observed Sharpe
    reflects a true positive, given `n_trials` alternatives were tested.

    Returns a value in [0, 1]. Interpretation:
    - > 0.95: strong evidence (rare in practice with multiple trials)
    - > 0.50: weak evidence
    - ~0.50: indistinguishable from noise
    - < 0.50: worse than the expected max under the null

    Reference: Bailey & López de Prado (2014).

    `returns` is per-period, not annualized. `n_trials` >= 1; with
    n_trials=1 the function degenerates to the Probabilistic Sharpe
    Ratio (PSR) against zero.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    arr = np.asarray(returns, dtype=np.float64)
    t = len(arr)
    if t < 2:
        return 0.5

    std = float(arr.std(ddof=1))
    if std < 1e-12 or not np.isfinite(std):
        return 0.5

    sr = float(arr.mean() / std)
    if not np.isfinite(sr):
        return 0.5

    skew_val = float(skew(arr))
    kurt_val = float(kurtosis(arr, fisher=False))
    if not (np.isfinite(skew_val) and np.isfinite(kurt_val)):
        return 0.5

    # Variance of the SR estimator, per Lo (2002)
    var_sr = (1 + 0.5 * sr**2 - skew_val * sr + ((kurt_val - 3) / 4) * sr**2) / (t - 1)
    if var_sr <= 0 or not np.isfinite(var_sr):
        var_sr = 1.0 / (t - 1)

    sr0 = 0.0 if n_trials == 1 else np.sqrt(var_sr) * _expected_max_sharpe_under_null(n_trials)

    denom_sq = 1 - skew_val * sr + ((kurt_val - 1) / 4) * sr**2
    denom = np.sqrt(max(denom_sq, 1e-12))
    z = (sr - sr0) * np.sqrt(t - 1) / denom
    if not np.isfinite(z):
        return 0.5
    return float(norm.cdf(z))


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_subsamples: int = 16,
) -> float:
    """Probability of Backtest Overfitting via CSCV.

    `returns_matrix` is (T, N_strategies): one column per strategy.
    Returns the fraction of combinations where the best in-sample
    strategy underperforms the median out-of-sample. Values:
    - 0.0: no evidence of overfitting (dominant strategy wins both IS/OOS)
    - 0.5: indistinguishable from random
    - > 0.5: overfitting likely
    - 1.0: pure overfitting (best IS is worst OOS every time)

    Reference: Bailey et al. (2017).

    With fewer than 2 strategies or fewer than 2*n_subsamples
    observations, returns 0.0 (cannot detect).
    """
    mat = np.asarray(returns_matrix, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    t, n = mat.shape
    if n < 2 or n_subsamples * 2 > t:
        return 0.0

    # Force even number of sub-samples
    s = n_subsamples if n_subsamples % 2 == 0 else n_subsamples - 1
    chunk_size = t // s
    if chunk_size < 2:
        return 0.0
    chunks = [mat[i * chunk_size : (i + 1) * chunk_size] for i in range(s)]
    n_combinations = 0
    n_overfit = 0
    half = s // 2

    for is_idx in combinations(range(s), half):
        is_set = set(is_idx)
        oos_idx = [i for i in range(s) if i not in is_set]
        is_data = np.concatenate([chunks[i] for i in is_idx], axis=0)
        oos_data = np.concatenate([chunks[i] for i in oos_idx], axis=0)

        is_mean = is_data.mean(axis=0)
        is_std = is_data.std(axis=0, ddof=1)
        is_std = np.where(is_std > 0, is_std, 1e-12)
        is_sharpe = is_mean / is_std
        best_idx = int(np.argmax(is_sharpe))

        oos_mean = oos_data.mean(axis=0)
        oos_std = oos_data.std(axis=0, ddof=1)
        oos_std = np.where(oos_std > 0, oos_std, 1e-12)
        oos_sharpe = oos_mean / oos_std

        # Rank of the best IS strategy in OOS (1 = worst)
        rank = int(np.sum(oos_sharpe < oos_sharpe[best_idx])) + 1
        if rank >= n:
            logit = 10.0  # best IS is best OOS (impossible overfit)
        elif rank <= 1:
            logit = -10.0
        else:
            logit = float(np.log(rank / (n - rank + 1)))
        if logit <= 0:
            n_overfit += 1
        n_combinations += 1

    return n_overfit / n_combinations if n_combinations > 0 else 0.0


def minimum_backtest_length(
    n_trials: int,
    target_sharpe_annualized: float,
) -> float:
    """Minimum backtest length (in years) needed for a given annualized
    Sharpe to be distinguishable from the expected max under N trials.

    If `target_sharpe <= E[max SR_N]`, no length suffices. Returns
    `inf` in that case.

    Reference: Bailey et al. (2014).
    """
    if target_sharpe_annualized <= 0:
        return float("inf")
    if n_trials < 2:
        return (1.0 / target_sharpe_annualized) ** 2
    e_max = _expected_max_sharpe_under_null(n_trials)
    if target_sharpe_annualized <= e_max:
        return float("inf")
    return float((e_max / target_sharpe_annualized) ** 2)


def block_bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_bootstrap: int = 1000,
    block_size: int | None = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the per-period Sharpe using
    the moving-block bootstrap (preserves autocorrelation).

    Returns (lower, upper) at (1-alpha) level. If `block_size` is None,
    uses sqrt(T) rounded down (rule of thumb).

    The interval is per-period, not annualized. Multiply by sqrt(ppy)
    for annualized version.
    """
    arr = np.asarray(returns, dtype=np.float64)
    t = len(arr)
    if t < 10:
        return (0.0, 0.0)
    if block_size is None:
        block_size = max(int(np.sqrt(t)), 1)
    if block_size >= t:
        block_size = max(t - 1, 1)

    rng = np.random.default_rng(seed)
    n_blocks = max(t // block_size, 1)
    max_start = t - block_size
    sharpes = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([arr[s : s + block_size] for s in starts])
        std = float(sample.std(ddof=1))
        if std > 0:
            sharpes[i] = float(sample.mean() / std)
        else:
            sharpes[i] = 0.0

    lower = float(np.quantile(sharpes, alpha / 2))
    upper = float(np.quantile(sharpes, 1 - alpha / 2))
    return (lower, upper)


def significance_summary(
    returns: np.ndarray,
    n_trials: int,
    periods_per_year: int,
    annualized_sharpe: float,
) -> dict:
    """Convenience wrapper: all four tests at once.

    - `returns`: per-period returns
    - `n_trials`: number of alternatives tested (for DSR)
    - `periods_per_year`: for annualizing the block bootstrap CI
    - `annualized_sharpe`: the observed annualized Sharpe (for MinBTL)
    """
    dsr = deflated_sharpe_ratio(returns, n_trials)
    ci_per_period = block_bootstrap_sharpe_ci(returns)
    ci_annualized = (
        ci_per_period[0] * np.sqrt(periods_per_year),
        ci_per_period[1] * np.sqrt(periods_per_year),
    )
    min_btl_years = minimum_backtest_length(n_trials, annualized_sharpe)
    return {
        "dsr": dsr,
        "sharpe_ci_annualized": ci_annualized,
        "min_btl_years": min_btl_years,
        "n_trials": n_trials,
    }
