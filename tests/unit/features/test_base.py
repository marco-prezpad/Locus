"""
test_base.py

FeatureRegistry: known families, unknown family, duplicate names,
FeatureSpec validation.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

import pytest

from locus.features import FeatureRegistry, FeatureSpec
from locus.features.pipeline import FeaturePipeline


def test_registry_builds_known_families() -> None:
    specs = [
        FeatureSpec(name="return_1", family="return", params={"n": 1}),
        FeatureSpec(name="ma_distance_20", family="ma_distance", params={"window": 20}),
    ]
    features = FeatureRegistry(specs).build()
    assert [f.name for f in features] == ["return_1", "ma_distance_20"]


def test_registry_unknown_family_raises() -> None:
    specs = [FeatureSpec(name="x", family="unknown")]
    with pytest.raises(ValueError, match="unknown feature family"):
        FeatureRegistry(specs).build()


def test_feature_spec_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        FeatureSpec(name="", family="return")


def test_feature_spec_rejects_empty_family() -> None:
    with pytest.raises(ValueError):
        FeatureSpec(name="x", family="")


def test_pipeline_rejects_duplicate_names() -> None:
    specs = [
        FeatureSpec(name="return_1", family="return", params={"n": 1}),
        FeatureSpec(name="return_1", family="return", params={"n": 3}),
    ]
    features = FeatureRegistry(specs).build()
    with pytest.raises(ValueError, match="duplicate feature names"):
        FeaturePipeline(features)


def test_pipeline_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FeaturePipeline([])
