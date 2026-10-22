"""
base.py

The Feature Protocol and the config-driven FeatureRegistry. A Feature
is a stateless transform: given a DataFrame with BAR_SCHEMA columns, it
returns the same DataFrame with one additional column named `self.name`.

The registry translates a list of FeatureSpec (from YAML) into Feature
instances. Families are registered in _FACTORIES; adding a new family
is one line.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import polars as pl

from locus.infrastructure.logging import LoggerFactory


class Feature(Protocol):
    """A stateless feature transform.

    Contract:
    - Input df has BAR_SCHEMA columns (asset_id, timestamp, open, high,
      low, close, volume).
    - Output df has all input columns plus one new column named
      `self.name`.
    - The feature is BACKWARD-LOOKING: at row t, its value depends only
      on data at timestamps <= t within the same asset. The leakage
      tests enforce this.
    - The feature may produce nulls for the initial rows of each asset
      (insufficient history). Downstream (F4/F5) drops them.
    """

    @property
    def name(self) -> str: ...

    def compute(self, df: pl.DataFrame) -> pl.DataFrame: ...


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """A feature specification as parsed from config."""

    name: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.family:
            raise ValueError("family must be non-empty")


class FeatureRegistry:
    """Build Feature instances from a list of FeatureSpec."""

    def __init__(self, specs: list[FeatureSpec]) -> None:
        self._specs = specs
        self._log = LoggerFactory.get("features.registry")

    def build(self) -> list[Feature]:
        """Instantiate all features. Fails at construction if a family
        is unknown or params are invalid, so misconfigured runs fail
        fast."""
        features: list[Feature] = []
        for spec in self._specs:
            factory = _FACTORIES.get(spec.family)
            if factory is None:
                raise ValueError(f"unknown feature family: {spec.family!r}")
            feature = factory(name=spec.name, **spec.params)
            features.append(feature)
        self._log.debug(
            "features_built",
            extra={"count": len(features), "families": [s.family for s in self._specs]},
        )
        return features

    def feature_names(self) -> list[str]:
        return [s.name for s in self._specs]


# Registry populated in each feature module via `register_feature`.
_FACTORIES: dict[str, Callable[..., Feature]] = {}


def register_feature(family: str) -> Callable[[Callable[..., Feature]], Callable[..., Feature]]:
    """Decorator that registers a feature factory for a given family
    name. Used by each feature module to self-register on import.
    """

    def decorator(factory: Callable[..., Feature]) -> Callable[..., Feature]:
        if family in _FACTORIES:
            raise ValueError(f"family already registered: {family!r}")
        _FACTORIES[family] = factory
        return factory

    return decorator
