"""
config.py

Experiment configuration loaded from YAML (plan v3.1 P0).

The schema follows Anexo B of the plan. Fields marked optional have
sensible defaults matching the B5 experiment; override them per
experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class CostScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fee_bps: float = Field(ge=0)
    spread_bps: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)
    impact_bps: float = Field(ge=0)


class CostsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    central: CostScenarioConfig
    pessimistic: CostScenarioConfig
    optimistic: CostScenarioConfig


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_folds: int = Field(ge=2)
    purge_hours: int = Field(ge=0, default=0)
    embargo_hours: int = Field(ge=0, default=0)


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    window: int | None = None


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    k: int = Field(ge=1, default=1)
    gross_exposure: float = Field(gt=0, default=1.0)
    k_sweep: list[int] = Field(default_factory=list)


class CriteriaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    e1_sharpe_above_baseline: float | None = None
    e2_dsr_p_max: float | None = 0.05
    e3_folds_positive_min: float | None = 0.70
    e4_regimes_btc_non_negative: bool | None = None
    e5_controls_abs_max: float | None = 0.20
    e6_concentration_max: float | None = 0.40
    e7_max_weight: float | None = 0.25
    e8_pessimistic_sharpe_min: float | None = 0.0
    e9_drawdown_max: float | None = 0.25
    e10_capacity_degradation_max: float | None = 0.30


class AbandonmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sharpe_below_baseline: bool = True
    dsr_p_above: float = 0.10
    folds_positive_below: float = 0.50
    control_abs_above: float = 0.20


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    thesis: str
    plan_version: str = "v3.1"
    hypothesis: str
    universe: str = "U10"

    feature: FeatureConfig
    strategy: StrategyConfig

    horizon_hours: int = Field(gt=0)
    rebalance_hours: int = Field(gt=0)
    annualization_days: int = Field(gt=0, default=365)

    walk_forward: WalkForwardConfig
    costs: CostsConfig
    cost_scenario: str = "central"
    baseline: str = "buy_and_hold_btc"

    criteria: CriteriaConfig = Field(default_factory=CriteriaConfig)
    abandonment: AbandonmentConfig = Field(default_factory=AbandonmentConfig)
    n_trials_before: int = Field(ge=0, default=0)

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def to_yaml_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
