"""
test_config.py

Verifies the configuration loader merges YAML with environment variables
correctly, enforces the signal-execution contract invariants, and loads
the three pre-registered cost scenarios with the expected ordering
(optimistic < central < pessimistic).

Author: Marco Pérez Padilla
Date:   20-10-2026
"""

from __future__ import annotations

import pytest

from locus.infrastructure.config import (
    Config,
    load_config,
    load_cost_scenario,
)


def test_load_config_from_yaml() -> None:
    config = load_config("configs/base.yaml")
    assert isinstance(config, Config)
    assert config.app.name == "locus"
    assert config.app.environment == "local"
    assert config.model.default_horizon_hours == 24
    assert config.portfolio.max_weight_per_asset == 0.10
    assert config.risk.max_daily_loss_pct == 0.02
    assert config.backtest.fee_bps == 10.0
    assert config.backtest.impact_k == 1.0


def test_signal_execution_contract_invariants() -> None:
    config = load_config("configs/base.yaml")
    sec = config.signal_execution
    assert sec.feature_compute_time == "bar_close_T"
    assert sec.earliest_execution_time == "bar_open_T+1"
    assert sec.execution_price == "bar_open_T+1"
    assert sec.intermediate_signals_used_for_execution is False
    assert sec.earliest_execution_time != sec.feature_compute_time


def test_config_defaults_when_yaml_missing() -> None:
    config = load_config("configs/does_not_exist.yaml")
    assert config.app.name == "locus"


def test_config_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP__LOG_LEVEL", "DEBUG")
    config = load_config("configs/base.yaml")
    assert config.app.log_level == "DEBUG"


def test_cost_scenarios_load() -> None:
    for name in ("optimistic", "central", "pessimistic"):
        costs = load_cost_scenario(name)  # type: ignore[arg-type]
        assert "fee_bps" in costs
        assert "impact_k" in costs
        assert costs["fee_bps"] > 0


def test_cost_scenarios_ordering() -> None:
    opt = load_cost_scenario("optimistic")
    cen = load_cost_scenario("central")
    pes = load_cost_scenario("pessimistic")
    assert opt["fee_bps"] < cen["fee_bps"] < pes["fee_bps"]
    assert opt["impact_k"] < cen["impact_k"] < pes["impact_k"]


def test_cost_scenario_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_cost_scenario("does_not_exist")  # type: ignore[arg-type]
