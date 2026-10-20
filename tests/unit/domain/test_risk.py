"""
test_risk.py

Verifies RiskDecision coupling.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

import pytest

from locus.domain import RiskDecision, RiskViolation, ViolationType
from locus.domain.types import AssetId


def _violation() -> RiskViolation:
    return RiskViolation(
        violation_type=ViolationType.POSITION_SIZE,
        asset_id=AssetId("BTC"),
        limit=0.1,
        observed=0.2,
        message="BTC weight 0.20 exceeds limit 0.10",
    )


def test_approval_without_violations() -> None:
    d = RiskDecision(
        approved=True,
        adjusted_weights={AssetId("BTC"): 0.1},
    )
    assert d.approved and not d.kill_switch


def test_rejection_requires_violation() -> None:
    with pytest.raises(ValueError):
        RiskDecision(approved=False)


def test_kill_switch_cannot_coexist_with_approval() -> None:
    with pytest.raises(ValueError):
        RiskDecision(approved=True, kill_switch=True)


def test_rejection_with_violation_ok() -> None:
    d = RiskDecision(approved=False, violations=[_violation()])
    assert d.violations[0].violation_type == ViolationType.POSITION_SIZE


def test_negative_adjusted_weight_rejected() -> None:
    with pytest.raises(ValueError):
        RiskDecision(approved=True, adjusted_weights={AssetId("BTC"): -0.1})


def test_violation_requires_message() -> None:
    with pytest.raises(ValueError):
        RiskViolation(
            violation_type=ViolationType.POSITION_SIZE,
            asset_id=None,
            limit=0.1,
            observed=0.2,
            message="",
        )
