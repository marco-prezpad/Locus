"""End-to-end runner test on synthetic data (P0)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from locus.experiments.runner import run_experiment


def _make_data(root: Path) -> None:
    import numpy as np

    processed = root / "processed"
    processed.mkdir(parents=True)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    n_hours = 2000
    assets = ("BTC", "ETH", "SOL")
    rng = np.random.default_rng(42)
    bar_rows = []
    funding_rows = []
    for asset in assets:
        for i in range(n_hours):
            ts = base + timedelta(hours=i)
            price = 100.0 + 0.01 * i
            bar_rows.append(
                {
                    "asset_id": asset,
                    "timestamp": ts,
                    "open": price,
                    "high": price + 1,
                    "low": price - 1,
                    "close": price,
                    "volume": 1000.0,
                }
            )
            if i % 8 == 0:
                funding_rows.append(
                    {
                        "asset_id": asset,
                        "venue": "binance_futures",
                        "symbol": f"{asset}USDT",
                        "timestamp": ts,
                        "funding_rate": float(rng.normal(0.0001, 0.0005)),
                    }
                )
    pl.DataFrame(bar_rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    ).write_parquet(processed / "bars.parquet")
    pl.DataFrame(funding_rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    ).write_parquet(processed / "funding.parquet")


def _make_config(tmp_path: Path) -> Path:
    body = """
id: "TEST-20261026-01"
thesis: "T1"
hypothesis: "synthetic"
feature: {name: "funding_zscore_30d", window: 21}
strategy:
  name: "FundingLongShortTopK"
  k: 2
  gross_exposure: 1.0
  k_sweep: [1, 2, 3]
horizon_hours: 24
rebalance_hours: 24
walk_forward: {n_folds: 2}
costs:
  central: {fee_bps: 10, spread_bps: 5, slippage_bps: 5, impact_bps: 1}
  pessimistic: {fee_bps: 20, spread_bps: 10, slippage_bps: 10, impact_bps: 2}
  optimistic: {fee_bps: 5, spread_bps: 2, slippage_bps: 2, impact_bps: 0.5}
baseline: "buy_and_hold_btc"
n_trials_before: 0
"""
    p = tmp_path / "exp.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_run_experiment_creates_report_and_registers(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_data(data_root)
    config = _make_config(tmp_path)
    registry = tmp_path / "trials" / "registry.jsonl"
    reports = tmp_path / "reports"

    result = run_experiment(
        config_path=config,
        registry_path=registry,
        reports_dir=reports,
        data_root=data_root,
        end_date=None,
        register=True,
        verify_reproducible=False,
    )

    assert result.config_id == "TEST-20261026-01"
    assert result.report_path.exists()
    assert registry.exists()
    assert result.validation.verdict in ("positive", "negative", "inconclusive")


def test_run_experiment_no_register(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_data(data_root)
    config = _make_config(tmp_path)
    registry = tmp_path / "trials" / "registry.jsonl"

    run_experiment(
        config_path=config,
        registry_path=registry,
        reports_dir=tmp_path / "reports",
        data_root=data_root,
        end_date=None,
        register=False,
        verify_reproducible=False,
    )

    assert not registry.exists()


def test_reproducibility_diff_below_1e_9(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_data(data_root)
    config = _make_config(tmp_path)

    result = run_experiment(
        config_path=config,
        registry_path=tmp_path / "trials" / "registry.jsonl",
        reports_dir=tmp_path / "reports",
        data_root=data_root,
        end_date=None,
        register=False,
        verify_reproducible=True,
    )

    assert result.reproducibility_diff is not None
    assert result.reproducibility_diff < 1e-9, (
        f"reproducibility diff = {result.reproducibility_diff:.3e} >= 1e-9"
    )
