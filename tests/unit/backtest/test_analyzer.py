"""
test_analyzer.py

report_to_row and reports_table: column presence, value flow, empty
handling.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

import numpy as np

from locus.backtest.analyzer import report_to_row, reports_table
from locus.backtest.events import BacktestReport
from locus.evaluation.metrics import PerformanceMetrics


def _report(
    name: str = "test",
    capital: float = 10_000.0,
    sharpe: float = 0.5,
    total_return: float = 0.15,
    n_executed: int = 100,
    n_generated: int = 120,
    n_skipped_min: int = 20,
) -> BacktestReport:
    return BacktestReport(
        strategy_name=name,
        capital=capital,
        equity_curve=np.array([capital, capital * (1 + total_return)]),
        returns=np.array([total_return]),
        metrics=PerformanceMetrics(
            total_return=total_return,
            cagr=total_return,
            volatility=0.3,
            sharpe=sharpe,
            sortino=sharpe * 1.2,
            max_drawdown=-0.2,
            calmar=total_return / 0.2,
            hit_rate=0.55,
            turnover=0.1,
            n_periods=100,
        ),
        n_orders_generated=n_generated,
        n_orders_executed=n_executed,
        n_orders_skipped_min=n_skipped_min,
        total_cost_quote=capital * 0.01,
        total_cost_bps=100.0,
    )


def test_report_to_row_has_all_columns() -> None:
    row = report_to_row(_report())
    expected = {
        "strategy",
        "capital",
        "n_periods",
        "total_return",
        "cagr",
        "volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "hit_rate",
        "turnover",
        "n_orders_generated",
        "n_orders_executed",
        "n_orders_skipped_min",
        "execution_rate",
        "total_cost_bps",
    }
    assert set(row.keys()) == expected


def test_report_to_row_values() -> None:
    row = report_to_row(_report(sharpe=0.42, total_return=0.08))
    assert row["strategy"] == "test"
    assert row["capital"] == 10_000.0
    assert row["sharpe"] == 0.42
    assert row["total_return"] == 0.08


def test_report_to_row_execution_rate() -> None:
    row = report_to_row(_report(n_executed=80, n_generated=100))
    assert row["execution_rate"] == 0.8


def test_report_to_row_zero_generated() -> None:
    row = report_to_row(_report(n_executed=0, n_generated=0))
    assert row["execution_rate"] == 0.0


def test_reports_table_empty() -> None:
    df = reports_table([])
    assert df.height == 0


def test_reports_table_single_row() -> None:
    df = reports_table([_report(name="A", sharpe=0.5)])
    assert df.height == 1
    assert df["strategy"][0] == "A"
    assert df["sharpe"][0] == 0.5


def test_reports_table_multiple_rows() -> None:
    reports = [
        _report(name="A", capital=1_000.0, sharpe=0.3),
        _report(name="B", capital=10_000.0, sharpe=0.6),
        _report(name="C", capital=100_000.0, sharpe=0.1),
    ]
    df = reports_table(reports)
    assert df.height == 3
    assert df["strategy"].to_list() == ["A", "B", "C"]
    assert df["capital"].to_list() == [1_000.0, 10_000.0, 100_000.0]
