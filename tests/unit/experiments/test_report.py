"""Tests for the Anexo C report generation (P0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from locus.evaluation.validation import (
    ValidationReport,
    compute_dsr,
    compute_pbo,
)
from locus.experiments.config import ExperimentConfig
from locus.experiments.registry import Trial
from locus.experiments.report import generate_report


def _experiment(tmp_path: Path) -> ExperimentConfig:
    body = """
id: "TEST-20261026-01"
thesis: "T1"
hypothesis: "test hypothesis"
feature: {name: "f", window: 90}
strategy: {name: "S", k: 2, gross_exposure: 1.0}
horizon_hours: 24
rebalance_hours: 24
walk_forward: {n_folds: 3}
costs:
  central: {fee_bps: 10, spread_bps: 5, slippage_bps: 5, impact_bps: 1}
  pessimistic: {fee_bps: 20, spread_bps: 10, slippage_bps: 10, impact_bps: 2}
  optimistic: {fee_bps: 5, spread_bps: 2, slippage_bps: 2, impact_bps: 0.5}
"""
    p = tmp_path / "exp.yaml"
    p.write_text(body, encoding="utf-8")
    return ExperimentConfig.from_yaml(p)


def _trial() -> Trial:
    return Trial(
        id="TEST-20261026-01",
        date="2026-10-26",
        thesis="T1",
        hypothesis="test",
        config_hash="sha256:abc",
        data_hash="sha256:def",
        commit="abc1234",
        metrics={"sharpe": 0.5},
        state="candidate",
        description="test",
    )


def _validation() -> ValidationReport:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, 200)
    rng.normal(0.0, 0.01, 200)
    matrix = np.column_stack([returns, rng.normal(0, 0.01, 200)])
    dsr = compute_dsr(returns, n_trials=10, periods_per_year=252)
    pbo = compute_pbo(matrix, s_blocks=8)
    return ValidationReport(
        sharpe=0.5,
        sharpe_ci_low=0.2,
        sharpe_ci_high=0.8,
        dsr=dsr,
        pbo=pbo,
        paired_p_vs_baseline=0.03,
        paired_observed_diff=0.3,
        n_trials=10,
        verdict="positive",
        rationale="all criteria met",
    )


def test_report_created(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    generate_report(
        experiment=_experiment(tmp_path),
        trial=_trial(),
        validation=_validation(),
        baseline_sharpe=0.2,
        output_path=out,
    )
    assert out.exists()


def test_report_contains_required_sections(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    generate_report(
        experiment=_experiment(tmp_path),
        trial=_trial(),
        validation=_validation(),
        baseline_sharpe=0.2,
        output_path=out,
    )
    text = out.read_text(encoding="utf-8")
    for section in [
        "# Experiment report",
        "## Hypothesis",
        "## Configuration",
        "## Results",
        "## Criteria",
        "## Verdict",
        "## Next step per plan",
        "## Known limitations",
    ]:
        assert section in text, f"missing section: {section}"


def test_verdict_is_uppercase(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    generate_report(
        experiment=_experiment(tmp_path),
        trial=_trial(),
        validation=_validation(),
        baseline_sharpe=0.2,
        output_path=out,
    )
    text = out.read_text(encoding="utf-8")
    assert "**POSITIVE**" in text
