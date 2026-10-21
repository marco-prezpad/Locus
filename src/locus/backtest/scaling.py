"""
scaling.py

CapitalScalingExperiment: run one strategy across multiple capital
scenarios with the same cost model, and produce a comparison table.

The point: show that the same strategy is unexecutable at $1, partially
at $10-$100, and representative at $1k+. This is the honest answer to
"¿funciona a mi escala?".

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from locus.backtest.analyzer import reports_table
from locus.backtest.broker import SimulatedBroker
from locus.backtest.config import load_backtest_config
from locus.backtest.engine import BacktestEngine
from locus.backtest.events import BacktestReport
from locus.baselines.base import BaselineStrategy
from locus.infrastructure.logging import LoggerFactory

CapitalName = Literal["1", "10", "100", "1k", "10k", "100k", "1m"]
CostScenario = Literal["optimistic", "central", "pessimistic"]


class CapitalScalingExperiment:
    def __init__(
        self,
        capital_names: list[CapitalName] | None = None,
        cost_scenario: CostScenario = "central",
    ) -> None:
        self._names: list[CapitalName] = capital_names or [
            "1",
            "10",
            "100",
            "1k",
            "10k",
            "100k",
            "1m",
        ]
        self._cost: CostScenario = cost_scenario
        self._log = LoggerFactory.get("backtest.scaling")

    def run(
        self,
        strategy: BaselineStrategy,
        ohlcv: pl.DataFrame,
        rebalance_freq_bars: int = 24,
    ) -> tuple[dict[CapitalName, BacktestReport], pl.DataFrame]:
        weights = strategy.generate_weights(ohlcv, rebalance_freq_bars)

        reports: dict[CapitalName, BacktestReport] = {}
        for name in self._names:
            config = load_backtest_config(
                strategy_name=strategy.name,
                capital_name=name,
                cost_scenario=self._cost,
                rebalance_freq_bars=rebalance_freq_bars,
            )
            engine = BacktestEngine(config, SimulatedBroker(config))
            report = engine.run(weights, ohlcv)
            reports[name] = report

        table = reports_table(list(reports.values()))
        return reports, table
