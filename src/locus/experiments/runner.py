"""
runner.py

Run an experiment end-to-end (plan v3.1 P0).

Flow: load config → hash config and data → load bars and funding →
run walk-forward on the strategy → run walk-forward on the baseline →
call canonical validation (DSR, PBO, paired diff, CI) → append to
registry (unless reproduction) → generate Anexo C report.

Optional `verify_reproducible`: runs the strategy twice and reports the
max absolute difference between per-period returns. This is the P0 exit
criterion (plan §9): diff < 1e-9.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from locus.backtest.config import BacktestConfig
from locus.backtest.costs import (
    FeeModel,
    MarketImpactModel,
    SlippageModel,
    SpreadModel,
    TransactionCostModel,
)
from locus.baselines.strategies import (
    BuyAndHoldBTC,
    EqualWeightUniverse,
    FundingLongShortTopK,
)
from locus.evaluation.validation import ValidationReport, run_validation
from locus.evaluation.walk_forward import WalkForwardBaselineEvaluator
from locus.experiments.config import ExperimentConfig
from locus.experiments.registry import (
    Trial,
    hash_config,
    hash_data,
)
from locus.experiments.registry import (
    append as registry_append,
)
from locus.experiments.registry import (
    count as registry_count,
)
from locus.experiments.report import generate_report
from locus.features.funding import (
    FundingTrendFeature,
    FundingXSRankFeature,
    FundingXSZScoreFeature,
    FundingZScoreFeature,
)

BAR_COLS = {"asset_id", "timestamp", "open", "high", "low", "close", "volume"}
FUNDING_COLS = {"asset_id", "timestamp", "funding_rate"}

LOT_SIZES: dict[str, float] = {
    "BTC": 0.00001,
    "ETH": 0.0001,
    "SOL": 0.001,
    "BNB": 0.001,
    "XRP": 0.1,
    "ADA": 1.0,
    "DOGE": 1.0,
    "AVAX": 0.001,
    "DOT": 0.01,
    "LINK": 0.01,
}


@dataclass(frozen=True, slots=True)
class RunResult:
    config_id: str
    trial: Trial
    validation: ValidationReport
    baseline_sharpe: float
    n_trials_used: int
    report_path: Path
    reproducibility_diff: float | None


# ---------------------------------------------------------------------------
# Config → engine primitives
# ---------------------------------------------------------------------------


def _cost_model(experiment: ExperimentConfig) -> TransactionCostModel:
    scenario = getattr(experiment.costs, experiment.cost_scenario)
    return TransactionCostModel(
        fee=FeeModel(scenario.fee_bps),
        spread=SpreadModel(scenario.spread_bps),
        slippage=SlippageModel(scenario.slippage_bps),
        impact=MarketImpactModel(scenario.impact_bps),
    )


def _build_backtest_config(
    experiment: ExperimentConfig,
    suffix: str,
) -> BacktestConfig:
    return BacktestConfig(
        strategy_name=f"{experiment.id}_{suffix}",
        capital=10_000.0,
        currency="USDT",
        cost_model=_cost_model(experiment),
        min_order_notional=5.0,
        default_lot_size=1e-8,
        lot_sizes=dict(LOT_SIZES),
        bars_per_day=24,
        rebalance_freq_bars=experiment.rebalance_hours,
        annualization_days=experiment.annualization_days,
        apply_funding=True,
    )


def _build_strategy(experiment: ExperimentConfig, k: int):
    if experiment.strategy.name == "FundingLongShortTopK":
        return FundingLongShortTopK(
            feature=experiment.feature.name,
            k=k,
            gross_exposure=experiment.strategy.gross_exposure,
            horizon_bars=experiment.horizon_hours,
            rebalance_bars=experiment.rebalance_hours,
        )
    raise SystemExit(f"Unknown strategy {experiment.strategy.name!r}")


def _build_baseline(name: str):
    if name == "buy_and_hold_btc":
        return BuyAndHoldBTC()
    if name == "equal_weight":
        return EqualWeightUniverse()
    raise SystemExit(f"Unknown baseline {name!r}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _has_cols(path: Path, required: set[str]) -> bool:
    try:
        return required.issubset(set(pl.read_parquet_schema(path).keys()))
    except Exception:
        return False


def _discover(root: Path, required: set[str]) -> list[Path]:
    return sorted(p for p in root.rglob("*.parquet") if _has_cols(p, required))


def _load_data(
    data_root: Path,
    end_date: str | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    bars_files = _discover(data_root, BAR_COLS)
    funding_files = _discover(data_root, FUNDING_COLS)
    if not bars_files or not funding_files:
        raise SystemExit(f"No bars or funding parquet under {data_root}")

    bars = pl.scan_parquet(bars_files).collect().sort(["asset_id", "timestamp"])
    funding = pl.scan_parquet(funding_files).collect().sort(["asset_id", "timestamp"])

    if end_date:
        cutoff = datetime.fromisoformat(end_date)
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=UTC)
        bars = bars.filter(pl.col("timestamp") <= cutoff)
        funding = funding.filter(pl.col("timestamp") <= cutoff)

    return bars, funding


def _align_common(
    bars: pl.DataFrame,
    funding: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    n_assets = bars["asset_id"].n_unique()
    common = (
        bars.group_by("timestamp")
        .agg(pl.col("asset_id").n_unique().alias("n"))
        .filter(pl.col("n") == n_assets)["timestamp"]
        .sort()
        .to_list()
    )
    if not common:
        raise SystemExit("No common timestamp across all assets.")
    lo, hi = common[0], common[-1]
    bars = bars.filter(pl.col("timestamp").is_in(common))
    funding = funding.filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") <= hi))
    return bars, funding


def _compute_feature(
    funding: pl.DataFrame,
    name: str,
    window: int | None,
) -> pl.DataFrame:
    if name == "funding_xs_rank":
        return FundingXSRankFeature(name=name, source="funding_rate").compute(funding)
    if name == "funding_xs_zscore":
        return FundingXSZScoreFeature(name=name, source="funding_rate").compute(funding)
    if name == "funding_trend_7d":
        return FundingTrendFeature(name=name, window=window or 21).compute(funding)
    return FundingZScoreFeature(name=name, window=window or 90).compute(funding)


def _attach_feature(
    bars: pl.DataFrame,
    funding: pl.DataFrame,
    name: str,
    window: int | None,
) -> pl.DataFrame:
    feat = _compute_feature(funding, name, window).select(["asset_id", "timestamp", name])
    return (
        bars.join(feat, on=["asset_id", "timestamp"], how="left")
        .sort(["asset_id", "timestamp"])
        .with_columns(pl.col(name).forward_fill().over("asset_id"))
    )


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


def _run_strategy(
    experiment: ExperimentConfig,
    bars: pl.DataFrame,
    funding: pl.DataFrame,
    k: int,
) -> np.ndarray:
    strategy = _build_strategy(experiment, k)
    cfg = _build_backtest_config(experiment, f"k{k}")
    wf = WalkForwardBaselineEvaluator(cfg, n_folds=experiment.walk_forward.n_folds)
    result = wf.run(
        strategy,
        bars,
        rebalance_freq_bars=experiment.rebalance_hours,
        funding=funding,
    )
    return result.returns


def _run_baseline(
    experiment: ExperimentConfig,
    bars: pl.DataFrame,
    funding: pl.DataFrame,
) -> np.ndarray:
    baseline = _build_baseline(experiment.baseline)
    cfg = _build_backtest_config(experiment, "baseline")
    wf = WalkForwardBaselineEvaluator(cfg, n_folds=experiment.walk_forward.n_folds)
    result = wf.run(
        baseline,
        bars,
        rebalance_freq_bars=experiment.rebalance_hours,
        funding=funding,
    )
    return result.returns


def _sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    sd = returns.std(ddof=1)
    if sd == 0:
        return 0.0
    return float((returns.mean() / sd) * np.sqrt(periods_per_year))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_experiment(
    config_path: Path,
    *,
    registry_path: Path = Path("trials/registry.jsonl"),
    reports_dir: Path = Path("reports"),
    data_root: Path = Path("data"),
    end_date: str | None = "2026-10-01",
    register: bool = True,
    verify_reproducible: bool = False,
) -> RunResult:
    """Run one experiment end-to-end and return the outcome."""
    experiment = ExperimentConfig.from_yaml(config_path)

    if len(experiment.strategy.k_sweep) < 2:
        raise SystemExit(
            "experiment.strategy.k_sweep must have at least 2 entries for "
            "PBO (Bailey et al. 2015 CSCV). Add two or more k values to "
            "the YAML config under strategy.k_sweep."
        )

    cfg_hash = hash_config(experiment.to_yaml_dict())
    data_hash = hash_data([data_root])
    commit = _git_commit()

    bars, funding = _load_data(data_root, end_date)
    bars = _attach_feature(bars, funding, experiment.feature.name, experiment.feature.window)
    bars, funding = _align_common(bars, funding)

    periods_per_year = int((24 * experiment.annualization_days) / experiment.rebalance_hours)

    # Main strategy run
    main_returns = _run_strategy(experiment, bars, funding, experiment.strategy.k)

    # k-sweep → PBO matrix
    matrixes = [_run_strategy(experiment, bars, funding, k) for k in experiment.strategy.k_sweep]
    n_min = min(len(m) for m in matrixes)
    returns_matrix = np.column_stack([m[:n_min] for m in matrixes])

    # Baseline
    baseline_returns = _run_baseline(experiment, bars, funding)
    baseline_sharpe = _sharpe(baseline_returns, periods_per_year)

    n_pair = min(len(main_returns), len(baseline_returns))
    main_aligned = main_returns[:n_pair]
    baseline_aligned = baseline_returns[:n_pair]

    n_trials = registry_count(registry_path) + (1 if register else 0)

    validation = run_validation(
        returns=main_aligned,
        returns_matrix=returns_matrix,
        baseline_returns=baseline_aligned,
        n_trials=max(n_trials, 1),
        periods_per_year=periods_per_year,
    )

    trial = Trial(
        id=experiment.id,
        date=datetime.now(UTC).strftime("%Y-%m-%d"),
        thesis=experiment.thesis,
        hypothesis=experiment.hypothesis,
        config_hash=cfg_hash,
        data_hash=data_hash,
        commit=commit,
        metrics={
            "sharpe": float(validation.sharpe),
            "sharpe_vectorized": None,
            "return": float(np.prod(1.0 + main_aligned) - 1.0),
            "baseline_sharpe": baseline_sharpe,
            "dsr": float(validation.dsr.dsr),
            "pbo": float(validation.pbo.pbo),
            "paired_p": float(validation.paired_p_vs_baseline),
        },
        state="candidate" if validation.verdict == "positive" else "discarded",
        description=validation.rationale,
    )

    if register:
        try:
            registry_append(trial, registry_path)
        except ValueError as exc:
            if "already exists" in str(exc):
                pass
            else:
                raise

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{experiment.id}.md"
    generate_report(
        experiment=experiment,
        trial=trial,
        validation=validation,
        baseline_sharpe=baseline_sharpe,
        output_path=report_path,
    )

    reproducibility_diff: float | None = None
    if verify_reproducible:
        second = _run_strategy(experiment, bars, funding, experiment.strategy.k)
        n_repro = min(len(main_returns), len(second))
        reproducibility_diff = float(np.max(np.abs(main_returns[:n_repro] - second[:n_repro])))

    return RunResult(
        config_id=experiment.id,
        trial=trial,
        validation=validation,
        baseline_sharpe=baseline_sharpe,
        n_trials_used=n_trials,
        report_path=report_path,
        reproducibility_diff=reproducibility_diff,
    )
