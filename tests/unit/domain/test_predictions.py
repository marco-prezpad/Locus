"""
test_predictions.py

Verifies Prediction and OpportunityScore invariants.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from locus.domain import OpportunityScore, Prediction
from locus.domain.types import AssetId, Timestamp


def _ts() -> Timestamp:
    return Timestamp(datetime(2026, 1, 1, tzinfo=UTC))


def _prediction(**overrides: object) -> Prediction:
    defaults = {
        "asset_id": AssetId("BTC"),
        "timestamp": _ts(),
        "horizon": timedelta(hours=24),
        "expected_return": 0.01,
        "expected_risk": 0.02,
        "probability_positive": 0.6,
        "confidence": 0.7,
        "model_id": "m1",
    }
    defaults.update(overrides)
    return Prediction(**defaults)  # type: ignore[arg-type]


def test_prediction_valid() -> None:
    p = _prediction()
    assert p.quantiles is None


def test_prediction_rejects_non_positive_horizon() -> None:
    with pytest.raises(ValueError):
        _prediction(horizon=timedelta(0))


def test_prediction_rejects_probability_out_of_range() -> None:
    with pytest.raises(ValueError):
        _prediction(probability_positive=1.5)


def test_prediction_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValueError):
        _prediction(confidence=-0.1)


def test_prediction_accepts_valid_quantiles() -> None:
    p = _prediction(quantiles={0.1: -0.03, 0.5: 0.01, 0.9: 0.05})
    assert p.quantiles[0.5] == pytest.approx(0.01)


def test_prediction_rejects_bad_quantile_key() -> None:
    with pytest.raises(ValueError):
        _prediction(quantiles={1.5: 0.0})


def test_opportunity_eligible_without_reason() -> None:
    s = OpportunityScore(
        asset_id=AssetId("BTC"),
        timestamp=_ts(),
        score=1.0,
        components={"expected_return": 0.01, "cost": 0.001},
        eligible=True,
    )
    assert s.eligible and s.reason is None


def test_opportunity_ineligible_requires_reason() -> None:
    with pytest.raises(ValueError):
        OpportunityScore(
            asset_id=AssetId("BTC"),
            timestamp=_ts(),
            score=0.0,
            components={},
            eligible=False,
        )


def test_opportunity_eligible_must_not_carry_reason() -> None:
    with pytest.raises(ValueError):
        OpportunityScore(
            asset_id=AssetId("BTC"),
            timestamp=_ts(),
            score=1.0,
            components={},
            eligible=True,
            reason="but",
        )
