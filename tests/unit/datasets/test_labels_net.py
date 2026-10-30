"""Tests for net-of-costs and offset labels (P4)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.datasets.labels import LabelBuilder
from locus.datasets.types import CostAssumption, LabelSpec


def _bars(prices: list[float], asset: str = "BTC") -> pl.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "asset_id": asset,
                "timestamp": base + timedelta(hours=i),
                "open": p * 0.99,  # open differs from close for offset tests
                "high": p,
                "low": p,
                "close": p,
                "volume": 1.0,
            }
            for i, p in enumerate(prices)
        ]
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def _cost() -> CostAssumption:
    # round_trip = 2 * (10 + 5/2 + 5 + 1) = 2 * 18.5 = 37 bps = 0.0037
    return CostAssumption(
        fee_bps=10.0,
        spread_bps=5.0,
        slippage_bps=5.0,
        impact_bps=1.0,
    )


def test_round_trip_bps_formula() -> None:
    assert _cost().round_trip_bps == pytest.approx(37.0)


def test_from_config_experiment_costs() -> None:
    import tempfile
    from pathlib import Path

    from locus.experiments.config import ExperimentConfig

    body = """
id: "TEST-20261030-01"
thesis: "T1"
hypothesis: "test"
feature: {name: "f"}
strategy: {name: "S", k: 1}
horizon_hours: 24
rebalance_hours: 24
walk_forward: {n_folds: 2}
costs:
  central: {fee_bps: 10, spread_bps: 5, slippage_bps: 5, impact_bps: 1}
  pessimistic: {fee_bps: 20, spread_bps: 10, slippage_bps: 10, impact_bps: 2}
  optimistic: {fee_bps: 5, spread_bps: 2, slippage_bps: 2, impact_bps: 0.5}
"""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "exp.yaml"
        p.write_text(body, encoding="utf-8")
        cfg = ExperimentConfig.from_yaml(p)
        c = CostAssumption.from_config(cfg.costs, "central")
        assert c.round_trip_bps == pytest.approx(37.0)
        cp = CostAssumption.from_config(cfg.costs, "pessimistic")
        assert cp.round_trip_bps == pytest.approx(2 * (20 + 5 + 10 + 2))


# --- backward-compatible behaviour ---


def test_default_close_offset_zero_preserves_old_behavior() -> None:
    """No cost, close prices, offset 0 → same output as the old builder."""
    prices = [100.0, 101.0, 102.0, 103.0, 104.0]
    df = _bars(prices)
    lb = LabelBuilder(
        specs=[LabelSpec(name="future_return_2h", bars=2)],
        probability_positive_from="future_return_2h",
    )
    out = lb.compute(df)
    row0 = out.row(0, named=True)
    assert row0["future_return_2h"] == pytest.approx(
        prices[2] / prices[0] - 1.0,
        rel=1e-9,
    )
    assert "future_return_2h_net" not in out.columns
    assert row0["probability_positive"] == 1


# --- net of costs ---


def test_net_column_added_when_cost_provided() -> None:
    prices = [100.0] * 10
    df = _bars(prices)
    lb = LabelBuilder(
        specs=[LabelSpec(name="future_return_2h", bars=2)],
        cost_assumption=_cost(),
    )
    out = lb.compute(df)
    assert "future_return_2h" in out.columns
    assert "future_return_2h_net" in out.columns
    # Flat prices → gross 0 → net -0.0037
    row0 = out.row(0, named=True)
    assert row0["future_return_2h"] == pytest.approx(0.0, abs=1e-12)
    assert row0["future_return_2h_net"] == pytest.approx(-0.0037, rel=1e-9)


def test_net_signal_used_by_binary_label() -> None:
    """Flat prices → net negative → probability_positive = 0."""
    prices = [100.0] * 10
    df = _bars(prices)
    lb = LabelBuilder(
        specs=[LabelSpec(name="future_return_2h", bars=2)],
        probability_positive_from="future_return_2h_net",
        cost_assumption=_cost(),
    )
    out = lb.compute(df)
    valid = out.filter(pl.col("probability_positive").is_not_null())
    assert (valid["probability_positive"] == 0).all()


def test_net_positive_when_gross_beats_cost() -> None:
    prices = [100.0 * (1.05**i) for i in range(10)]
    df = _bars(prices)
    lb = LabelBuilder(
        specs=[LabelSpec(name="future_return_1h", bars=1)],
        probability_positive_from="future_return_1h_net",
        cost_assumption=_cost(),
    )
    out = lb.compute(df)
    valid = out.filter(pl.col("probability_positive").is_not_null())
    assert (valid["probability_positive"] == 1).all()


# --- custom threshold ---


def test_custom_threshold_u() -> None:
    prices = [100.0 * (1.05**i) for i in range(10)]
    df = _bars(prices)
    lb = LabelBuilder(
        specs=[LabelSpec(name="future_return_1h", bars=1)],
        probability_positive_from="future_return_1h_net",
        cost_assumption=_cost(),
        threshold_u=0.05,
    )
    out = lb.compute(df)
    valid = out.filter(pl.col("probability_positive").is_not_null())
    assert (valid["probability_positive"] == 0).all()


# --- entry offset and open price (plan §7.3) ---


def test_open_offset_matches_plan() -> None:
    """Plan §7.3: entry at open[t+1], exit at open[t+1+h]."""
    prices = [100.0, 101.0, 102.0, 103.0, 104.0]
    df = _bars(prices)  # open = price * 0.99
    lb = LabelBuilder(
        specs=[LabelSpec(name="future_return_2h", bars=2)],
        price_field="open",
        entry_offset_bars=1,
    )
    out = lb.compute(df)
    # row0: entry = open[1] = 101*0.99, exit = open[3] = 103*0.99
    row0 = out.row(0, named=True)
    expected = (103.0 * 0.99) / (101.0 * 0.99) - 1.0
    assert row0["future_return_2h"] == pytest.approx(expected, rel=1e-9)


def test_tail_nulls_with_offset() -> None:
    """With offset=1 and bars=h, last 1+h rows are null."""
    df = _bars([100.0] * 10)
    lb = LabelBuilder(
        specs=[LabelSpec(name="future_return_3h", bars=3)],
        price_field="open",
        entry_offset_bars=1,
    )
    out = lb.compute(df)
    tail = out.tail(4)  # 1 + 3
    assert tail["future_return_3h"].is_null().all()


# --- per asset ---


def test_per_asset_independent() -> None:
    df = pl.concat(
        [
            _bars([100.0, 101.0, 102.0, 103.0, 104.0], "A"),
            _bars([10.0, 10.1, 10.2, 10.3, 10.4], "B"),
        ]
    )
    lb = LabelBuilder(
        specs=[LabelSpec(name="future_return_2h", bars=2)],
    )
    out = lb.compute(df)
    ts0 = df["timestamp"][0]
    row_a = out.filter((pl.col("asset_id") == "A") & (pl.col("timestamp") == ts0)).row(
        0, named=True
    )
    row_b = out.filter((pl.col("asset_id") == "B") & (pl.col("timestamp") == ts0)).row(
        0, named=True
    )
    assert row_a["future_return_2h"] == pytest.approx(102.0 / 100.0 - 1, rel=1e-9)
    assert row_b["future_return_2h"] == pytest.approx(10.2 / 10.0 - 1, rel=1e-9)


# --- validation ---


def test_bad_probability_source() -> None:
    with pytest.raises(ValueError, match="probability_positive_from"):
        LabelBuilder(
            specs=[LabelSpec(name="future_return_2h", bars=2)],
            probability_positive_from="future_return_2h_net",  # not available without cost
        )


def test_probability_source_valid_with_cost() -> None:
    """`future_return_2h_net` is valid when cost is provided."""
    LabelBuilder(
        specs=[LabelSpec(name="future_return_2h", bars=2)],
        probability_positive_from="future_return_2h_net",
        cost_assumption=_cost(),
    )


def test_cost_assumption_rejects_negative() -> None:
    with pytest.raises(ValueError, match="fee_bps"):
        CostAssumption(fee_bps=-1.0, spread_bps=5, slippage_bps=5, impact_bps=1)


def test_bad_price_field() -> None:
    with pytest.raises(ValueError, match="price_field"):
        LabelBuilder(
            specs=[LabelSpec(name="f", bars=1)],
            price_field="mid",  # type: ignore[arg-type]
        )


def test_negative_offset_rejected() -> None:
    with pytest.raises(ValueError, match="entry_offset_bars"):
        LabelBuilder(
            specs=[LabelSpec(name="f", bars=1)],
            entry_offset_bars=-1,
        )
