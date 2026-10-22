"""
test_experiments.py

ExperimentRegistry: CRUD, counters, persistence, final-one-shot rule.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from pathlib import Path

import pytest

from locus.evaluation.experiments import (
    ExperimentRecord,
    ExperimentRegistry,
)


def test_empty_registry(tmp_path: Path) -> None:
    reg = ExperimentRegistry(tmp_path / "registry.json")
    assert len(reg) == 0
    assert reg.count() == 0


def test_add_exploration(tmp_path: Path) -> None:
    reg = ExperimentRegistry(tmp_path / "registry.json")
    rec = reg.add(purpose="exploration", strategy="xgb", metrics={"sharpe": 0.5})
    assert isinstance(rec, ExperimentRecord)
    assert len(reg) == 1
    assert reg.count("exploration") == 1


def test_n_trials_excludes_exploration(tmp_path: Path) -> None:
    reg = ExperimentRegistry(tmp_path / "registry.json")
    reg.add(purpose="exploration", strategy="a")
    reg.add(purpose="selection", strategy="b")
    reg.add(purpose="selection", strategy="c")
    assert reg.n_trials() == 2


def test_n_trials_minimum_one(tmp_path: Path) -> None:
    reg = ExperimentRegistry(tmp_path / "registry.json")
    assert reg.n_trials() == 1


def test_final_one_shot_rule(tmp_path: Path) -> None:
    reg = ExperimentRegistry(tmp_path / "registry.json")
    reg.add(purpose="final", strategy="x")
    with pytest.raises(ValueError, match="only one 'final'"):
        reg.add(purpose="final", strategy="y")


def test_persistence(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    reg = ExperimentRegistry(path)
    reg.add(purpose="selection", strategy="a", metrics={"sharpe": 0.3})
    reg.add(purpose="selection", strategy="b", metrics={"sharpe": 0.5})

    reg2 = ExperimentRegistry(path)
    assert len(reg2) == 2
    assert reg2.n_trials() == 2
    assert reg2.all()[0].strategy == "a"


def test_invalid_purpose_raises(tmp_path: Path) -> None:
    reg = ExperimentRegistry(tmp_path / "registry.json")
    with pytest.raises(ValueError):
        reg.add(purpose="invalid", strategy="x")  # type: ignore[arg-type]
