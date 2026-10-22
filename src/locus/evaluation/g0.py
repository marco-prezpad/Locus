"""
g0.py

The G0 gate: pre-registered criterion, per-strategy report, and the
final decision.

Two criteria are pre-registered:

- `G0_CRITERION_V1`: the original F7 criterion (min_dsr=0.95, 2 years).
- `G0_CRITERION_V2`: calibrated for pivote A (weekly horizon, 5 years).
  See docs/governance.md §7.b for the calibration rationale.

Neither is modified after seeing results. A new criterion requires a
new research cycle.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from locus.evaluation.significance import (
    block_bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    minimum_backtest_length,
    probability_of_backtest_overfitting,
)
from locus.evaluation.walk_forward import WalkForwardStrategyResult
from locus.infrastructure.logging import LoggerFactory

Outcome = Literal["positive", "negative", "inconclusive"]


@dataclass(frozen=True, slots=True)
class G0Criterion:
    """Pre-registered criteria for gate G0.

    A strategy passes G0 if it satisfies ALL of:
    - `metrics.sharpe >= min_sharpe`
    - `dsr >= min_dsr`
    - `ci_low > 0`
    - `min_btl_years <= available_years`
    - `pbo < max_pbo` (across all strategies)
    - `sharpe > benchmark_sharpe + min_sharpe_vs_benchmark`
    """

    min_sharpe: float = 0.0
    min_dsr: float = 0.95
    available_years: float = 2.0
    max_pbo: float = 0.5
    benchmark_name: str = "buy_and_hold_btc"
    n_trials_for_dsr: int = 4
    min_sharpe_vs_benchmark: float = 0.0


# Pre-registered criterion for F7 (daily horizon, 2 years).
G0_CRITERION_V1 = G0Criterion(
    min_sharpe=0.0,
    min_dsr=0.95,
    available_years=2.0,
    max_pbo=0.5,
    benchmark_name="buy_and_hold_btc",
    n_trials_for_dsr=4,
    min_sharpe_vs_benchmark=0.0,
)

# Calibrated criterion for pivote A (weekly horizon, 5 years).
# The DSR 0.95 threshold requires Sharpe > 1.21 with 5y of data, which is
# not achievable with public technical features. See docs/governance.md
# §7.b for the full calibration rationale.
G0_CRITERION_V2 = G0Criterion(
    min_sharpe=0.0,
    min_dsr=0.80,
    available_years=5.0,
    max_pbo=0.5,
    benchmark_name="buy_and_hold_btc",
    n_trials_for_dsr=4,
    min_sharpe_vs_benchmark=0.10,
)


@dataclass(frozen=True, slots=True)
class StrategyG0Report:
    """G0 metrics for one strategy."""

    strategy_name: str
    sharpe: float
    total_return: float
    max_drawdown: float
    dsr: float
    ci_low: float
    ci_high: float
    min_btl_years: float
    passes: bool
    fail_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class G0Report:
    """Final G0 report."""

    outcome: Outcome
    benchmark_name: str
    benchmark_sharpe: float
    per_strategy: list[StrategyG0Report]
    pbo: float
    n_trials: int
    criterion: G0Criterion
    rationale: str


def _compute_strategy_report(
    result: WalkForwardStrategyResult,
    criterion: G0Criterion,
) -> StrategyG0Report:
    """Compute DSR, CI, MinBTL and pass/fail reasons for one strategy.

    The benchmark comparison (`sharpe > benchmark_sharpe + min_sharpe_vs_benchmark`)
    is applied in `build_g0_report`, because it needs the benchmark's
    Sharpe, which is only known after all strategies are computed.
    """
    returns = result.returns
    if len(returns) < 2:
        return StrategyG0Report(
            strategy_name=result.strategy_name,
            sharpe=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            dsr=0.5,
            ci_low=0.0,
            ci_high=0.0,
            min_btl_years=float("inf"),
            passes=False,
            fail_reasons=["insufficient_returns"],
        )

    sharpe = float(result.metrics.sharpe)
    dsr = deflated_sharpe_ratio(returns, n_trials=criterion.n_trials_for_dsr)
    ci_low_per, ci_high_per = block_bootstrap_sharpe_ci(returns)
    ci_low = ci_low_per * np.sqrt(result.periods_per_year)
    ci_high = ci_high_per * np.sqrt(result.periods_per_year)
    min_btl = minimum_backtest_length(criterion.n_trials_for_dsr, sharpe)

    fails: list[str] = []
    if sharpe < criterion.min_sharpe:
        fails.append(f"sharpe {sharpe:.3f} < {criterion.min_sharpe}")
    if dsr < criterion.min_dsr:
        fails.append(f"dsr {dsr:.3f} < {criterion.min_dsr}")
    if ci_low <= 0:
        fails.append(f"ci_low {ci_low:.3f} <= 0")
    if not np.isfinite(min_btl):
        fails.append("min_btl infinite")
    elif min_btl > criterion.available_years:
        fails.append(f"min_btl {min_btl:.2f}y > {criterion.available_years}y available")

    return StrategyG0Report(
        strategy_name=result.strategy_name,
        sharpe=sharpe,
        total_return=float(result.metrics.total_return),
        max_drawdown=float(result.metrics.max_drawdown),
        dsr=float(dsr),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        min_btl_years=float(min_btl),
        passes=len(fails) == 0,
        fail_reasons=fails,
    )


def build_g0_report(
    results: list[WalkForwardStrategyResult],
    criterion: G0Criterion = G0_CRITERION_V1,
) -> G0Report:
    """Build the G0 report from walk-forward results and the pre-
    registered criterion."""
    if not results:
        raise ValueError("results must be non-empty")

    log = LoggerFactory.get("evaluation.g0")
    per_strategy = [_compute_strategy_report(r, criterion) for r in results]

    # PBO across all strategies' concatenated returns
    min_len = (
        min(len(r.returns) for r in results if len(r.returns) > 0)
        if any(len(r.returns) > 0 for r in results)
        else 0
    )
    pbo = 0.0
    if min_len >= 16:
        matrix = np.column_stack([r.returns[:min_len] for r in results if len(r.returns) > 0])
        pbo = probability_of_backtest_overfitting(matrix, n_subsamples=16)

    benchmark = next(
        (s for s in per_strategy if s.strategy_name == criterion.benchmark_name),
        None,
    )
    if benchmark is None:
        raise ValueError(f"benchmark {criterion.benchmark_name!r} not in results")

    candidates = [s for s in per_strategy if s.strategy_name != criterion.benchmark_name]
    pbo_ok = pbo < criterion.max_pbo

    # An active strategy must (a) pass all individual criteria AND
    # (b) beat the benchmark by at least `min_sharpe_vs_benchmark`.
    passing_active = [
        s
        for s in candidates
        if s.passes and s.sharpe > benchmark.sharpe + criterion.min_sharpe_vs_benchmark
    ]

    if pbo_ok and passing_active:
        winners = [s.strategy_name for s in passing_active]
        outcome: Outcome = "positive"
        rationale = (
            f"Positive G0: {len(winners)} active strateg(ies) pass all "
            f"criteria and beat benchmark Sharpe {benchmark.sharpe:.3f} "
            f"by >= {criterion.min_sharpe_vs_benchmark}: {winners}. "
            f"PBO={pbo:.3f} < {criterion.max_pbo}."
        )
    elif not pbo_ok:
        outcome = "negative"
        rationale = (
            f"Negative G0: PBO={pbo:.3f} >= {criterion.max_pbo}; ranking "
            f"is unstable, no strategy can be trusted out of sample. "
            f"Benchmark Sharpe={benchmark.sharpe:.3f}."
        )
    else:
        outcome = "negative"
        rationale = (
            f"Negative G0: no active strategy passes all criteria and "
            f"beats benchmark Sharpe {benchmark.sharpe:.3f} by >= "
            f"{criterion.min_sharpe_vs_benchmark}. "
            f"Benchmark itself: DSR={benchmark.dsr:.3f}, "
            f"CI=[{benchmark.ci_low:+.3f}, {benchmark.ci_high:+.3f}], "
            f"MinBTL={'inf' if not np.isfinite(benchmark.min_btl_years) else f'{benchmark.min_btl_years:.2f}y'}. "
            f"PBO={pbo:.3f}. Pivot per docs/governance.md."
        )

    log.debug(
        "g0_report_built",
        extra={
            "outcome": outcome,
            "n_strategies": len(per_strategy),
            "pbo": round(pbo, 3),
            "benchmark_sharpe": round(benchmark.sharpe, 3),
            "criterion": criterion.min_dsr,
        },
    )

    return G0Report(
        outcome=outcome,
        benchmark_name=benchmark.strategy_name,
        benchmark_sharpe=benchmark.sharpe,
        per_strategy=per_strategy,
        pbo=pbo,
        n_trials=criterion.n_trials_for_dsr,
        criterion=criterion,
        rationale=rationale,
    )
