"""
labels.py

LabelBuilder computes forward-looking labels per asset:
future_return_<name>[t] = close[t + bars] / close[t] - 1.

Labels use future data by design (that's what a label is). Leakage is
prevented by the splitter (purge + walk-forward), not by the label
builder. The critical invariant: label[t] uses close[t] as the entry
price, so features[t] (which use close up to t) do not overlap with
the label's future window except at t itself.

Also adds a binary `probability_positive` column derived from the
configured source label. Calibration of the model that predicts it
happens in F9/F10, not here.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

import polars as pl

from locus.datasets.types import LabelSpec
from locus.infrastructure.logging import LoggerFactory

_PROBABILITY_POSITIVE = "probability_positive"


class LabelBuilder:
    """Compute forward-return labels and a binary positive-return label."""

    def __init__(
        self,
        specs: list[LabelSpec],
        probability_positive_from: str | None = None,
    ) -> None:
        if not specs:
            raise ValueError("specs must be non-empty")
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate label names: {names}")
        if probability_positive_from is not None and probability_positive_from not in names:
            raise ValueError(
                f"probability_positive_from {probability_positive_from!r} "
                f"is not among the label names: {names}"
            )
        self._specs = specs
        self._prob_from = probability_positive_from
        self._log = LoggerFactory.get("datasets.labels")

    @property
    def label_names(self) -> list[str]:
        names = [s.name for s in self._specs]
        if self._prob_from is not None:
            names = names + [_PROBABILITY_POSITIVE]
        return names

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add all labels to df, per asset, sorted by (asset_id, timestamp).

        The last `max(bars)` rows of each asset will have null labels
        (insufficient future data); downstream drops them.
        """
        result = df.sort(["asset_id", "timestamp"])
        for spec in self._specs:
            result = result.with_columns(
                (pl.col("close").shift(-spec.bars).over("asset_id") / pl.col("close") - 1).alias(
                    spec.name
                )
            )
        if self._prob_from is not None:
            result = result.with_columns(
                (pl.col(self._prob_from) > 0).cast(pl.Int8).alias(_PROBABILITY_POSITIVE)
            )
        self._log.info(
            "labels_computed",
            extra={
                "rows": result.height,
                "labels": self.label_names,
            },
        )
        return result
