"""
metrics.py

Performance metrics for strategy evaluation. All annualizations use
`periods_per_year` (e.g., 365 for daily rebalancing, 8760 for hourly).

Convention: `returns` is an array of per-rebalance returns. The equity
curve is built by compounding: equity[t] = equity[t-1] * (1 + returns[t]).

Risk-free rate is annualized; it is converted to per-period by dividing
by `periods_per_year`.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """All metrics for one strategy evaluation."""

    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    turnover: float
    n_periods: int

    @classmethod
    def empty(cls) -> PerformanceMetrics:
        return cls(
            total_return=0.0,
            cagr=0.0,
            volatility=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            calmar=0.0,
            hit_rate=0.0,
            turnover=0.0,
            n_periods=0,
        )


def compute_metrics(
    returns: np.ndarray,
    periods_per_year: int,
    risk_free_rate_annual: float = 0.0,
    turnover: float = 0.0,
    eps: float = 1e-12,
) -> PerformanceMetrics:
    """Compute performance metrics from a per-rebalance returns array.

    `eps` guards against floating-point noise in std (e.g., a truly
    constant series whose variance rounds to 1e-17). Without it, Sharpe
    explodes to ~1e17 instead of being reported as 0.
    """
    n = len(returns)
    if n == 0:
        return PerformanceMetrics.empty()

    equity = np.concatenate([[1.0], np.cumprod(1 + returns)])
    total_return = float(equity[-1] - 1.0)

    years = n / periods_per_year
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else 0.0

    volatility = float(np.std(returns, ddof=1) * np.sqrt(periods_per_year)) if n >= 2 else 0.0

    rf_per_period = risk_free_rate_annual / periods_per_year
    excess = returns - rf_per_period

    if n >= 2:
        std_excess = float(np.std(excess, ddof=1))
        if std_excess > eps:
            sharpe = float(np.mean(excess) / std_excess * np.sqrt(periods_per_year))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    downside = excess[excess < 0]
    if len(downside) > 1:
        std_down = float(np.std(downside, ddof=1))
        if std_down > eps:
            sortino = float(np.mean(excess) / std_down * np.sqrt(periods_per_year))
        else:
            sortino = 0.0
    else:
        sortino = 0.0

    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    max_drawdown = float(np.min(drawdowns))

    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < -eps else 0.0

    hit_rate = float(np.mean(returns > 0))

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        calmar=calmar,
        hit_rate=hit_rate,
        turnover=turnover,
        n_periods=n,
    )
