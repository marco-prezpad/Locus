"""
types.py

Dataset-layer types: label specs, split results, fold descriptions.
These are the contracts between LabelBuilder, splitters,
TrainingDatasetBuilder, and DatasetValidator.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl


@dataclass(frozen=True, slots=True)
class LabelSpec:
    """One label: `future_return_<name>` over `bars` bars.

    Semantics: close[t + bars] / close[t] - 1, per asset. Uses future
    data (that's what labels do); the leakage guarantee is enforced by
    the splitter, not by the label builder.
    """

    name: str
    bars: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.bars <= 0:
            raise ValueError("bars must be positive")


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """A named temporal split of the dataset with uniqueness weights
    and effective sample size.

    `weights` has one row per sample in `df`, computed as 1/concurrency
    per asset over `horizon_bars`. `n_eff = sum(weights)`, i.e., the
    number of independent observations after accounting for overlap.
    """

    name: str
    df: pl.DataFrame
    weights: pl.Series
    n_eff: float
    start: datetime
    end: datetime

    @property
    def n_samples(self) -> int:
        return self.df.height


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One walk-forward fold: contiguous train/val/test windows with
    purge gaps between them."""

    fold: int
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit


@dataclass(frozen=True, slots=True)
class CostAssumption:
    """Fixed round-trip cost assumption for label netting (plan §7.3).

    Labels must reflect net-of-costs returns so that any model trained on
    them optimises against the executable target, not the gross one. The
    engine applies costs per side; for labels we use a fixed round-trip
    estimate derived from the same central scenario, because position size
    is unknown at label time (it depends on model weights).

    `round_trip_bps` = 2 * (fee + spread/2 + slippage + impact) per side.
    The spread is halved because you cross it once per side, not once per
    round trip (buy at ask, sell at bid → one spread crossed total).

    A `from_config` helper reads the central/pessimistic/optimistic
    scenario from `configs/base.yaml` so that labels and backtests agree
    by construction.
    """

    fee_bps: float
    spread_bps: float
    slippage_bps: float
    impact_bps: float

    def __post_init__(self) -> None:
        for name in ("fee_bps", "spread_bps", "slippage_bps", "impact_bps"):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"{name} must be >= 0, got {v}")

    @property
    def round_trip_bps(self) -> float:
        per_side = self.fee_bps + self.spread_bps / 2.0 + self.slippage_bps + self.impact_bps
        return 2.0 * per_side

    @classmethod
    def from_config(
        cls,
        config: Any,
        scenario: str = "central",
    ) -> CostAssumption:
        """Read cost fields from one of three shapes:

        1. `ExperimentConfig.costs` (has `.central`, `.pessimistic`, ...)
        2. `Config` (has `.backtest` which is a `BacktestConfig`)
        3. `BacktestConfig` directly (has `.fee_bps`, `.spread_bps`, ...)

        `scenario` is only used for shape 1. For shapes 2 and 3 the
        scenario concept does not apply (the values are already fixed).
        """
        # Shape 1: ExperimentConfig.costs
        if hasattr(config, "central"):
            sc = getattr(config, scenario)
            return cls(
                fee_bps=float(sc.fee_bps),
                spread_bps=float(sc.spread_bps),
                slippage_bps=float(sc.slippage_bps),
                impact_bps=float(sc.impact_bps),
            )
        # Shape 2: Config (unwrap .backtest)
        if hasattr(config, "backtest"):
            bt = config.backtest
            return cls(
                fee_bps=float(bt.fee_bps),
                spread_bps=float(bt.spread_bps),
                slippage_bps=float(bt.slippage_bps),
                impact_bps=float(bt.impact_bps),
            )
        # Shape 3: BacktestConfig directly
        if hasattr(config, "fee_bps"):
            return cls(
                fee_bps=float(config.fee_bps),
                spread_bps=float(config.spread_bps),
                slippage_bps=float(config.slippage_bps),
                impact_bps=float(
                    getattr(config, "impact_bps", None) or getattr(config, "impact_k", 1.0)
                ),
            )
        raise ValueError(f"Cannot extract CostAssumption from {type(config).__name__}")
