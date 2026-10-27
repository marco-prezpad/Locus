"""
report.py

Generate Anexo C experiment report as Markdown (plan v3.1 P0).

The report is the single artifact a reviewer reads to judge an experiment.
Fields match Anexo C of the plan. Empty fields are filled with explicit
"n/a" or "(see fold report)" so the reader never wonders whether something
was forgotten.
"""

from __future__ import annotations

from pathlib import Path

from locus.evaluation.validation import ValidationReport
from locus.experiments.config import ExperimentConfig
from locus.experiments.registry import Trial


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.4f}"
    return str(value)


def generate_report(
    experiment: ExperimentConfig,
    trial: Trial,
    validation: ValidationReport,
    baseline_sharpe: float,
    output_path: Path,
) -> None:
    """Write the Anexo C report for one experiment to `output_path`."""
    c = experiment.criteria
    lines: list[str] = []

    lines.append(f"# Experiment report: {experiment.id}")
    lines.append("")
    lines.append(f"**Thesis:** {experiment.thesis}")
    lines.append(f"**Plan version:** {experiment.plan_version}")
    lines.append(f"**Config hash:** `{trial.config_hash}`")
    lines.append(f"**Data hash:** `{trial.data_hash}`")
    lines.append(f"**Commit:** `{trial.commit}`")
    lines.append(f"**Trials before:** {experiment.n_trials_before}")
    lines.append(f"**Trials after:** {experiment.n_trials_before + 1}")
    lines.append("")

    lines.append("## Hypothesis")
    lines.append("")
    lines.append(experiment.hypothesis)
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    feature_desc = f"`{experiment.feature.name}`"
    if experiment.feature.window is not None:
        feature_desc += f" (window={experiment.feature.window})"
    lines.append(f"- Feature: {feature_desc}")
    lines.append(
        f"- Strategy: `{experiment.strategy.name}` "
        f"(k={experiment.strategy.k}, gross={experiment.strategy.gross_exposure})"
    )
    if experiment.strategy.k_sweep:
        lines.append(f"- k sweep (for PBO matrix): {experiment.strategy.k_sweep}")
    lines.append(f"- Horizon: {experiment.horizon_hours}h, rebalance {experiment.rebalance_hours}h")
    lines.append(
        f"- Walk-forward: {experiment.walk_forward.n_folds} folds, "
        f"purge {experiment.walk_forward.purge_hours}h, "
        f"embargo {experiment.walk_forward.embargo_hours}h"
    )
    lines.append(f"- Cost scenario: {experiment.cost_scenario}")
    lines.append(f"- Baseline: `{experiment.baseline}`")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append(f"- **Sharpe (net):** {validation.sharpe:+.4f}")
    lines.append(
        f"- **Sharpe CI 95%:** [{validation.sharpe_ci_low:+.4f}, {validation.sharpe_ci_high:+.4f}]"
    )
    lines.append(f"- **Baseline Sharpe:** {baseline_sharpe:+.4f}")
    lines.append(
        f"- **Paired diff vs baseline:** "
        f"{validation.paired_observed_diff:+.4f} "
        f"(p={validation.paired_p_vs_baseline:.4f})"
    )
    lines.append(
        f"- **DSR:** {validation.dsr.dsr:.4f} "
        f"(n_trials={validation.dsr.n_trials}, n_obs={validation.dsr.n_obs}, "
        f"threshold=0.95)"
    )
    lines.append(
        f"- **PBO:** {validation.pbo.pbo:.4f} "
        f"(s_blocks={validation.pbo.s_blocks}, "
        f"n_combinations={validation.pbo.n_combinations})"
    )
    lines.append("")

    lines.append("## Criteria (E1-E10)")
    lines.append("")
    lines.append("| Code | Criterion | Threshold | Value | Pass |")
    lines.append("|------|-----------|-----------|-------|------|")

    if c.e1_sharpe_above_baseline is not None:
        th = baseline_sharpe + c.e1_sharpe_above_baseline
        passed = validation.sharpe > th
        lines.append(
            f"| E1 | Sharpe > baseline + delta | {th:+.4f} | "
            f"{validation.sharpe:+.4f} | {'✓' if passed else '✗'} |"
        )
    if c.e2_dsr_p_max is not None:
        th = 1.0 - c.e2_dsr_p_max
        passed = validation.dsr.dsr > th
        lines.append(
            f"| E2 | DSR p < p_max | {th:.4f} | "
            f"{validation.dsr.dsr:.4f} | {'✓' if passed else '✗'} |"
        )
    if c.e5_controls_abs_max is not None:
        lines.append(
            f"| E5 | Controls abs Sharpe <= th | {c.e5_controls_abs_max:.2f} | (not run) | — |"
        )
    if c.e9_drawdown_max is not None:
        lines.append(
            f"| E9 | MaxDD <= threshold | -{c.e9_drawdown_max:.2f} | (see fold report) | — |"
        )
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{validation.verdict.upper()}**")
    lines.append("")
    lines.append(f"Rationale: {validation.rationale}")
    lines.append("")

    lines.append("## Next step per plan")
    lines.append("")
    if validation.verdict == "positive":
        lines.append(
            "Per plan §13: move to branch A. P12 (capacity), "
            "P6 (slippage calibrated with real trades), P15 (paper trading "
            "8 weeks)."
        )
    elif validation.verdict == "negative":
        lines.append(
            "Per plan §13: branch C. Run diagnosis D1-D4 (§8.1), then "
            "choose A1 (edge threshold), A2 (horizon), A3 (OHLCV features), "
            "A4 (microstructure, gated on P1.c), A5 (other market), or A6 "
            "(honest exit)."
        )
    else:
        lines.append(
            "Per plan §13: branch B. Apply A1 and/or A2 with the same pipeline and registry."
        )
    lines.append("")

    lines.append("## Known limitations")
    lines.append("")
    lines.append(f"- Universe: {experiment.universe} (fixed, subject to survivorship bias per P2).")
    lines.append("- History: ~2 years; Sharpe CI is wide (plan R3).")
    lines.append(
        "- Engine formal validation: P6 pending (slippage calibration with "
        "real trades, P1.c capture)."
    )
    lines.append(
        "- Walk-forward restarts capital per fold; aggregates are not "
        "comparable to a continuous equity curve (plan §6.1)."
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
