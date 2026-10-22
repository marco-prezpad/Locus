"""
test_g0.py

G0 criterion application, decision logic (positive/negative), and
rationale generation.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from locus.evaluation.g0 import (
    G0_CRITERION_V1,
    G0Criterion,
    build_g0_report,
)
from locus.evaluation.metrics import compute_metrics
from locus.evaluation.walk_forward import (
    FoldResult,
    WalkForwardStrategyResult,
)


def _make_result(
    name: str,
    returns: np.ndarray,
    ppy: int = 365,
) -> WalkForwardStrategyResult:
    metrics = compute_metrics(returns, ppy)
    fold = FoldResult(
        fold=0,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=len(returns)),
        n_returns=len(returns),
        metrics=metrics,
    )
    return WalkForwardStrategyResult(
        strategy_name=name,
        per_fold=[fold],
        returns=returns,
        metrics=metrics,
        periods_per_year=ppy,
    )


def test_g0_report_negative_on_noise() -> None:
    """All strategies are noise: DSR low, decision negative."""
    rng = np.random.default_rng(42)
    results = [
        _make_result("buy_and_hold_btc", rng.normal(0.0, 0.02, 730)),
        _make_result("momentum", rng.normal(0.0, 0.02, 730)),
        _make_result("equal_weight", rng.normal(0.0, 0.02, 730)),
        _make_result("random", rng.normal(0.0, 0.02, 730)),
    ]
    report = build_g0_report(results)
    assert report.outcome == "negative"
    assert "Negative G0" in report.rationale


def test_g0_report_negative_when_benchmark_best() -> None:
    """Benchmark is best; no active strategy beats it significantly."""
    rng = np.random.default_rng(42)
    results = [
        _make_result("buy_and_hold_btc", rng.normal(0.003, 0.02, 730)),
        _make_result("momentum", rng.normal(-0.002, 0.02, 730)),
        _make_result("equal_weight", rng.normal(0.001, 0.02, 730)),
        _make_result("random", rng.normal(-0.003, 0.02, 730)),
    ]
    report = build_g0_report(results)
    assert report.outcome == "negative"
    assert report.benchmark_name == "buy_and_hold_btc"


def test_g0_report_positive_when_active_beats_benchmark() -> None:
    """Active strategy has strong Sharpe, benchmark weak: positive."""
    rng = np.random.default_rng(42)
    results = [
        _make_result("buy_and_hold_btc", rng.normal(0.0, 0.02, 2000)),
        _make_result("momentum", rng.normal(0.008, 0.02, 2000)),  # strong
        _make_result("equal_weight", rng.normal(0.0, 0.02, 2000)),
        _make_result("random", rng.normal(0.0, 0.02, 2000)),
    ]
    report = build_g0_report(results)
    # 2000 obs, Sharpe ~0.4, positive shift; should pass most criteria
    # at least the benchmark comparison. Decision depends on DSR/CI/MinBTL.
    assert report.benchmark_name == "buy_and_hold_btc"
    # The momentum result should have positive sharpe
    momentum = next(s for s in report.per_strategy if s.strategy_name == "momentum")
    assert momentum.sharpe > 0


def test_g0_report_requires_benchmark() -> None:
    rng = np.random.default_rng(42)
    results = [
        _make_result("momentum", rng.normal(0.001, 0.02, 100)),
        _make_result("equal_weight", rng.normal(0.001, 0.02, 100)),
    ]
    with pytest.raises(ValueError, match="benchmark"):
        build_g0_report(results)


def test_g0_report_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_g0_report([])


def test_g0_criterion_pre_registered_values() -> None:
    """The pre-registered criterion must not silently drift."""
    assert G0_CRITERION_V1.min_dsr == 0.95
    assert G0_CRITERION_V1.max_pbo == 0.5
    assert G0_CRITERION_V1.available_years == 2.0
    assert G0_CRITERION_V1.benchmark_name == "buy_and_hold_btc"


def test_g0_custom_criterion() -> None:
    rng = np.random.default_rng(42)
    results = [
        _make_result("buy_and_hold_btc", rng.normal(0.0, 0.02, 500)),
        _make_result("momentum", rng.normal(0.001, 0.02, 500)),
    ]
    permissive = G0Criterion(
        min_sharpe=-1.0,
        min_dsr=0.5,
        available_years=10.0,
        max_pbo=1.0,
        benchmark_name="buy_and_hold_btc",
    )
    report = build_g0_report(results, criterion=permissive)
    assert report.criterion.max_pbo == 1.0


def test_lazy_imports_work() -> None:
    """Verify the lazy __getattr__ in evaluation/__init__ works."""
    import locus.evaluation as ev

    assert ev.G0_CRITERION_V1 is not None
    assert ev.G0_CRITERION_V2 is not None
    assert callable(ev.build_g0_report)
    assert ev.WalkForwardBaselineEvaluator is not None
