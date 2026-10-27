"""Tests for experiment YAML config (P0)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from locus.experiments.config import ExperimentConfig

_MINIMAL_YAML = """
id: "PB-20261025-13"
thesis: "PB"
hypothesis: "test"
feature: {name: "funding_zscore_30d", window: 90}
strategy: {name: "FundingLongShortTopK", k: 4}
horizon_hours: 168
rebalance_hours: 168
walk_forward: {n_folds: 3}
costs:
  central: {fee_bps: 10, spread_bps: 5, slippage_bps: 5, impact_bps: 1}
  pessimistic: {fee_bps: 20, spread_bps: 10, slippage_bps: 10, impact_bps: 2}
  optimistic: {fee_bps: 5, spread_bps: 2, slippage_bps: 2, impact_bps: 0.5}
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "exp.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_minimal_config(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(_write(tmp_path, _MINIMAL_YAML))
    assert cfg.id == "PB-20261025-13"
    assert cfg.strategy.k == 4
    assert cfg.walk_forward.n_folds == 3


def test_missing_required_field_fails(tmp_path: Path) -> None:
    body = _MINIMAL_YAML.replace('thesis: "PB"', "")
    with pytest.raises(ValidationError):
        ExperimentConfig.from_yaml(_write(tmp_path, body))


def test_unknown_field_fails(tmp_path: Path) -> None:
    body = _MINIMAL_YAML + '\nunknown_key: "x"\n'
    with pytest.raises(ValidationError, match="unknown_key"):
        ExperimentConfig.from_yaml(_write(tmp_path, body))


def test_k_sweep_defaults_to_empty(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(_write(tmp_path, _MINIMAL_YAML))
    assert cfg.strategy.k_sweep == []


def test_criteria_defaults(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(_write(tmp_path, _MINIMAL_YAML))
    assert cfg.criteria.e2_dsr_p_max == 0.05
    assert cfg.criteria.e3_folds_positive_min == 0.70
