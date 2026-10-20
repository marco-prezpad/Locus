"""
risk.py

Risk-management outputs: the decision the Risk Manager returns for a
given target portfolio, and the individual violations that triggered
adjustments.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from dataclasses import dataclass, field

from locus.domain.enums import ViolationType
from locus.domain.types import AssetId


@dataclass(frozen=True, slots=True)
class RiskViolation:
    """One detected breach of a risk limit."""

    violation_type: ViolationType
    asset_id: AssetId | None
    limit: float
    observed: float
    message: str

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError("message must be non-empty")


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """The Risk Manager's verdict on a proposed target portfolio.

    `approved=False` means the portfolio was rejected or adjusted;
    `adjusted_weights` carries the resulting weights. `kill_switch=True`
    means the system must halt execution entirely.
    """

    approved: bool
    adjusted_weights: dict[AssetId, float] = field(default_factory=dict)
    violations: list[RiskViolation] = field(default_factory=list)
    kill_switch: bool = False

    def __post_init__(self) -> None:
        if self.kill_switch and self.approved:
            raise ValueError("kill switch cannot coexist with approval")
        if not self.approved and not self.violations and not self.kill_switch:
            raise ValueError("rejected decision requires at least one violation or kill switch")
        if any(w < 0 for w in self.adjusted_weights.values()):
            raise ValueError("adjusted weights must be non-negative")
