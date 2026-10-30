"""
labels.py

LabelBuilder computes forward-looking labels per asset.

Gross labels (backward compatible, plan-agnostic):
    future_return_<name>[t] = price[t + offset + bars] / price[t + offset] - 1

Net labels (when cost_assumption is provided, plan §7.3):
    future_return_<name>_net[t] = future_return_<name>[t] - round_trip_bps/1e4

Binary label (when probability_positive_from is set):
    probability_positive[t] = 1 if <source>[t] > threshold_u else 0
    (null where source is null)

Entry offset and price field
----------------------------
- Backward-compatible defaults: `price_field="close"`, `entry_offset_bars=0`.
  Entry at close[t], exit at close[t+bars]. Preserves the existing
  TrainingDatasetBuilder behaviour without config changes.
- Plan §7.3 for T1: `price_field="open"`, `entry_offset_bars=1`.
  Entry at open[t+1], exit at open[t+1+bars]. Matches execution timing:
  features at close[t], execution at open[t+1].

Labels use future data by design (that's what a label is). Leakage is
prevented by the splitter (purge + embargo), not by the label builder.
The critical invariant: label[t] uses open[t+1] (or close[t]) as the
entry price, so features[t] (which use close up to t) do not overlap
with the label's future window.

Author: Marco Pérez Padilla
Date:   30-10-2026
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from locus.datasets.types import CostAssumption, LabelSpec
from locus.infrastructure.logging import LoggerFactory

_PROBABILITY_POSITIVE = "probability_positive"


class LabelBuilder:
    """Compute forward-return labels and a binary positive-return label."""

    def __init__(
        self,
        specs: list[LabelSpec],
        probability_positive_from: str | None = None,
        *,
        cost_assumption: CostAssumption | None = None,
        threshold_u: float = 0.0,
        price_field: Literal["close", "open"] = "close",
        entry_offset_bars: int = 0,
    ) -> None:
        if not specs:
            raise ValueError("specs must be non-empty")
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate label names: {names}")

        available_sources = names[:]
        if cost_assumption is not None:
            available_sources.extend(f"{n}_net" for n in names)

        if (
            probability_positive_from is not None
            and probability_positive_from not in available_sources
        ):
            raise ValueError(
                f"probability_positive_from {probability_positive_from!r} "
                f"is not among {available_sources}"
            )
        if price_field not in ("close", "open"):
            raise ValueError(f"price_field must be 'close' or 'open', got {price_field!r}")
        if entry_offset_bars < 0:
            raise ValueError("entry_offset_bars must be >= 0")

        self._specs = specs
        self._prob_from = probability_positive_from
        self._cost = cost_assumption
        self._threshold_u = threshold_u
        self._price_field = price_field
        self._offset = entry_offset_bars
        self._log = LoggerFactory.get("datasets.labels")

    @property
    def label_names(self) -> list[str]:
        names = [s.name for s in self._specs]
        if self._cost is not None:
            names = names + [f"{n}_net" for n in names[: len(self._specs)]]
        if self._prob_from is not None:
            names = names + [_PROBABILITY_POSITIVE]
        return names

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add all labels to df, per asset, sorted by (asset_id, timestamp).

        The last `offset + max(bars)` rows of each asset have null labels
        (insufficient future data); downstream drops them.
        """
        result = df.sort(["asset_id", "timestamp"])
        price = self._price_field

        if self._offset == 0:
            entry_expr = pl.col(price)
        else:
            entry_expr = pl.col(price).shift(-self._offset).over("asset_id")

        for spec in self._specs:
            exit_expr = pl.col(price).shift(-(self._offset + spec.bars)).over("asset_id")
            result = result.with_columns((exit_expr / entry_expr - 1).alias(spec.name))

        if self._cost is not None:
            cost = self._cost.round_trip_bps / 10_000.0
            for spec in self._specs:
                result = result.with_columns((pl.col(spec.name) - cost).alias(f"{spec.name}_net"))

        if self._prob_from is not None:
            result = result.with_columns(
                pl.when(pl.col(self._prob_from).is_not_null())
                .then((pl.col(self._prob_from) > self._threshold_u).cast(pl.Int8))
                .otherwise(None)
                .alias(_PROBABILITY_POSITIVE)
            )

        self._log.info(
            "labels_computed",
            extra={
                "rows": result.height,
                "labels": self.label_names,
                "price_field": price,
                "entry_offset_bars": self._offset,
                "net_of_costs": self._cost is not None,
            },
        )
        return result
