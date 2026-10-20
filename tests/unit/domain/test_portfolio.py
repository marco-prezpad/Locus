"""
test_portfolio.py

Verifies CapitalScenario and TargetPortfolio invariants, including the
weights-plus-cash constraint and the new `currency` field.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from locus.domain import (
    CapitalScenario,
    CostScenarioName,
    ImpactModel,
    TargetPortfolio,
)
from locus.domain.types import AssetId, Timestamp


def _ts() -> Timestamp:
    return Timestamp(datetime(2026, 1, 1, tzinfo=UTC))


def _scenario(**overrides: object) -> CapitalScenario:
    defaults = {
        "name": CostScenarioName.CENTRAL,
        "currency": "USDT",
        "capital": 10_000.0,
        "min_order_notional": 10.0,
        "fee_bps": 10.0,
        "spread_bps": 5.0,
        "slippage_bps": 5.0,
        "impact_model": ImpactModel.SQRT_VOLUME,
        "impact_k": 1.0,
        "fee_schedule_version": "binance-spot-2026-01",
    }
    defaults.update(overrides)
    return CapitalScenario(**defaults)  # type: ignore[arg-type]


def test_capital_scenario_valid() -> None:
    s = _scenario()
    assert s.impact_k == 1.0
    assert s.currency == "USDT"


def test_capital_scenario_accepts_other_currencies() -> None:
    for ccy in ("USD", "EUR", "BTC", "ETH"):
        s = _scenario(currency=ccy)
        assert s.currency == ccy


def test_capital_scenario_rejects_empty_currency() -> None:
    with pytest.raises(ValueError):
        _scenario(currency="")


def test_capital_scenario_rejects_lowercase_currency() -> None:
    with pytest.raises(ValueError):
        _scenario(currency="usdt")


def test_capital_scenario_rejects_non_alpha_currency() -> None:
    with pytest.raises(ValueError):
        _scenario(currency="US1T")


def test_capital_scenario_rejects_too_short_currency() -> None:
    with pytest.raises(ValueError):
        _scenario(currency="U")


def test_capital_scenario_rejects_too_long_currency() -> None:
    with pytest.raises(ValueError):
        _scenario(currency="USDTXX")


def test_capital_scenario_rejects_zero_capital() -> None:
    with pytest.raises(ValueError):
        _scenario(capital=0.0)


def test_capital_scenario_rejects_negative_cost() -> None:
    with pytest.raises(ValueError):
        _scenario(fee_bps=-1.0)


def test_target_portfolio_valid() -> None:
    tp = TargetPortfolio(
        timestamp=_ts(),
        weights={AssetId("BTC"): 0.5, AssetId("ETH"): 0.3},
        cash_weight=0.2,
        capital_scenario=_scenario(),
    )
    assert sum(tp.weights.values()) + tp.cash_weight == pytest.approx(1.0)
    assert tp.capital_scenario.currency == "USDT"


def test_target_portfolio_under_invested_ok() -> None:
    tp = TargetPortfolio(
        timestamp=_ts(),
        weights={AssetId("BTC"): 0.3},
        cash_weight=0.2,
        capital_scenario=_scenario(),
    )
    assert sum(tp.weights.values()) + tp.cash_weight == pytest.approx(0.5)


def test_target_portfolio_rejects_oversum() -> None:
    with pytest.raises(ValueError):
        TargetPortfolio(
            timestamp=_ts(),
            weights={AssetId("BTC"): 0.7, AssetId("ETH"): 0.5},
            cash_weight=0.2,
            capital_scenario=_scenario(),
        )


def test_target_portfolio_rejects_negative_weight() -> None:
    with pytest.raises(ValueError):
        TargetPortfolio(
            timestamp=_ts(),
            weights={AssetId("BTC"): -0.1},
            cash_weight=0.2,
            capital_scenario=_scenario(),
        )
