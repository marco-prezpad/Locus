"""
test_costs.py

Cost model correctness: individual components, composite, sqrt impact
scaling, cost scenario loading.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

import pytest

from locus.backtest.costs import (
    FeeModel,
    MarketImpactModel,
    SlippageModel,
    SpreadModel,
    TransactionCostModel,
    load_cost_model,
)


def test_fee_model_returns_fee() -> None:
    assert FeeModel(fee_bps=10.0).cost_bps(notional=1000.0) == 10.0


def test_fee_model_rejects_negative() -> None:
    with pytest.raises(ValueError):
        FeeModel(fee_bps=-1.0)


def test_spread_model_half_spread() -> None:
    # Half-spread per trade: spread_bps=5 → 2.5 bps per trade
    assert SpreadModel(spread_bps=5.0).cost_bps(notional=1000.0) == 2.5


def test_slippage_model_passthrough() -> None:
    assert SlippageModel(slippage_bps=5.0).cost_bps(notional=1000.0) == 5.0


def test_impact_zero_on_zero_notional() -> None:
    model = MarketImpactModel(impact_k=1.0)
    assert model.cost_bps(0.0, 1e9, 0.03) == 0.0


def test_impact_zero_on_zero_volume() -> None:
    model = MarketImpactModel(impact_k=1.0)
    assert model.cost_bps(1000.0, 0.0, 0.03) == 0.0


def test_impact_sqrt_scales_with_sqrt_of_notional() -> None:
    model = MarketImpactModel(impact_k=1.0)
    c1 = model.cost_bps(1_000.0, 1e9, 0.03)
    c4 = model.cost_bps(4_000.0, 1e9, 0.03)
    # 4x notional → 2x impact (sqrt scaling)
    assert c4 == pytest.approx(2.0 * c1)


def test_impact_linear_scales_linearly() -> None:
    model = MarketImpactModel(impact_k=1.0, model="linear")
    c1 = model.cost_bps(1_000.0, 1e9, 0.03)
    c4 = model.cost_bps(4_000.0, 1e9, 0.03)
    # 4x notional → 4x impact (linear scaling)
    assert c4 == pytest.approx(4.0 * c1)


def test_impact_none_is_zero() -> None:
    model = MarketImpactModel(impact_k=1.0, model="none")
    assert model.cost_bps(1e6, 1e9, 0.03) == 0.0


def test_impact_rejects_unknown_model() -> None:
    with pytest.raises(ValueError):
        MarketImpactModel(impact_k=1.0, model="quadratic")  # type: ignore[arg-type]


def test_composite_total_cost() -> None:
    model = TransactionCostModel(
        fee=FeeModel(fee_bps=10.0),
        spread=SpreadModel(spread_bps=5.0),
        slippage=SlippageModel(slippage_bps=5.0),
        impact=MarketImpactModel(impact_k=1.0),
    )
    # Central scenario on a small $10k order (impact tiny but non-zero)
    total = model.total_cost_bps(notional=10_000.0, daily_volume_quote=1e9, sigma_daily=0.03)
    breakdown = model.cost_breakdown(10_000.0, 1e9, 0.03)
    assert breakdown["fee"] == 10.0
    assert breakdown["spread"] == 2.5
    assert breakdown["slippage"] == 5.0
    assert breakdown["impact"] < 1.0  # tiny for $10k on $1B daily
    assert total == pytest.approx(sum(breakdown.values()))


def test_composite_impact_dominates_for_large_orders() -> None:
    model = TransactionCostModel(
        fee=FeeModel(fee_bps=10.0),
        spread=SpreadModel(spread_bps=5.0),
        slippage=SlippageModel(slippage_bps=5.0),
        impact=MarketImpactModel(impact_k=1.0),
    )
    small = model.cost_breakdown(10_000.0, 1e9, 0.03)
    large = model.cost_breakdown(10_000_000.0, 1e9, 0.03)
    # Impact grows ~sqrt(1000) ≈ 32x
    assert large["impact"] > 10 * small["impact"]
    # Total grows, driven by impact
    assert sum(large.values()) > sum(small.values())


def test_load_cost_model_optimistic() -> None:
    model = load_cost_model("optimistic")
    assert model.fee.fee_bps == 5.0
    assert model.spread.spread_bps == 2.0
    assert model.slippage.slippage_bps == 2.0
    assert model.impact.impact_k == 0.5


def test_load_cost_model_central() -> None:
    model = load_cost_model("central")
    assert model.fee.fee_bps == 10.0
    assert model.spread.spread_bps == 5.0
    assert model.slippage.slippage_bps == 5.0
    assert model.impact.impact_k == 1.0


def test_load_cost_model_pessimistic() -> None:
    model = load_cost_model("pessimistic")
    assert model.fee.fee_bps == 20.0
    assert model.spread.spread_bps == 10.0
    assert model.slippage.slippage_bps == 15.0
    assert model.impact.impact_k == 1.5


def test_load_cost_model_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_cost_model("does_not_exist")  # type: ignore[arg-type]
