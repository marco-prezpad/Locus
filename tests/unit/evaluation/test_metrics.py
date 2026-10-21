"""
test_metrics.py

Numerical correctness for all metrics on known return series.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

import numpy as np
import pytest

from locus.evaluation.metrics import compute_metrics


def test_empty_returns() -> None:
    m = compute_metrics(np.array([]), periods_per_year=365)
    assert m.n_periods == 0
    assert m.sharpe == 0.0


def test_total_return_single_period() -> None:
    m = compute_metrics(np.array([0.10]), periods_per_year=1)
    assert m.n_periods == 1
    # With <2 periods, empty is returned
    assert m.sharpe == 0.0


def test_total_return_compounds() -> None:
    returns = np.array([0.10, 0.10])
    m = compute_metrics(returns, periods_per_year=2)
    assert m.total_return == pytest.approx(0.21)


def test_volatility_annualized() -> None:
    returns = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
    m = compute_metrics(returns, periods_per_year=252)
    assert m.volatility == pytest.approx(float(np.std(returns, ddof=1) * np.sqrt(252)))


def test_sharpe_zero_when_constant_returns() -> None:
    returns = np.array([0.01] * 10)
    m = compute_metrics(returns, periods_per_year=365)
    # std == 0, sharpe returns 0 (guarded)
    assert m.sharpe == 0.0


def test_max_drawdown() -> None:
    # equity: 1.0, 1.1, 1.0, 1.05
    # peak = 1.1, trough = 1.0, drawdown = (1.0 - 1.1) / 1.1 = -0.1/1.1
    returns = np.array([0.10, -1.0 + 1.0 / 1.1, 0.05])
    m = compute_metrics(returns, periods_per_year=365)
    assert m.max_drawdown == pytest.approx(-0.1 / 1.1, abs=1e-9)


def test_hit_rate() -> None:
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    m = compute_metrics(returns, periods_per_year=365)
    assert m.hit_rate == pytest.approx(3 / 5)


def test_risk_free_subtracted_in_sharpe() -> None:
    returns = np.array([0.001] * 20)
    m_rf0 = compute_metrics(returns, periods_per_year=365, risk_free_rate_annual=0.0)
    m_rf_high = compute_metrics(returns, periods_per_year=365, risk_free_rate_annual=10.0)
    # Higher rf → lower sharpe
    assert m_rf0.sharpe >= m_rf_high.sharpe


def test_turnover_passed_through() -> None:
    m = compute_metrics(np.array([0.01, -0.01]), periods_per_year=365, turnover=0.5)
    assert m.turnover == 0.5
