"""
predictions.py

Model outputs and derived opportunity scores. A Prediction is what a
model produces; an OpportunityScore is what the decision engine derives
from it after applying costs, liquidity, and eligibility filters.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from locus.domain.types import AssetId, Timestamp


@dataclass(frozen=True, slots=True)
class Prediction:
    """A model's forecast for one asset over a specific horizon."""

    asset_id: AssetId
    timestamp: Timestamp
    horizon: timedelta
    expected_return: float
    expected_risk: float
    probability_positive: float | None
    confidence: float
    model_id: str
    quantiles: dict[float, float] | None = None
    metadata: dict[str, str] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.horizon <= timedelta(0):
            raise ValueError("horizon must be strictly positive")
        if self.expected_risk < 0:
            raise ValueError("expected_risk must be non-negative")
        if self.probability_positive is not None and not (0.0 <= self.probability_positive <= 1.0):
            raise ValueError("probability_positive must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.quantiles is not None:
            for q in self.quantiles:
                if not 0.0 <= q <= 1.0:
                    raise ValueError("quantile keys must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class OpportunityScore:
    """A comparable score across assets, with its component breakdown."""

    asset_id: AssetId
    timestamp: Timestamp
    score: float
    components: dict[str, float]
    eligible: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.eligible and not self.reason:
            raise ValueError("ineligible opportunity requires a reason")
        if self.eligible and self.reason is not None:
            raise ValueError("eligible opportunity must not carry a reason")
