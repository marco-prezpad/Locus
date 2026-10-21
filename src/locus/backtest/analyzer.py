"""
analyzer.py

Helpers to turn `BacktestReport` objects into comparison tables and
per-scenario summaries.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

import polars as pl

from locus.backtest.events import BacktestReport


def report_to_row(report: BacktestReport) -> dict:
    m = report.metrics
    return {
        "strategy": report.strategy_name,
        "capital": report.capital,
        "n_periods": m.n_periods,
        "total_return": m.total_return,
        "cagr": m.cagr,
        "volatility": m.volatility,
        "sharpe": m.sharpe,
        "sortino": m.sortino,
        "max_drawdown": m.max_drawdown,
        "calmar": m.calmar,
        "hit_rate": m.hit_rate,
        "turnover": m.turnover,
        "n_orders_generated": report.n_orders_generated,
        "n_orders_executed": report.n_orders_executed,
        "n_orders_skipped_min": report.n_orders_skipped_min,
        "execution_rate": report.execution_rate,
        "total_cost_bps": report.total_cost_bps,
    }


def reports_table(reports: list[BacktestReport]) -> pl.DataFrame:
    """Wide comparison table, one row per report."""
    rows = [report_to_row(r) for r in reports]
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)
