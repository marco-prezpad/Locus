"""
test_types.py

FxPair, FxRate invariants.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from locus.data.fx.types import FxPair, FxRate


def test_pair_parse() -> None:
    p = FxPair.parse("EURUSD")
    assert p.base == "EUR"
    assert p.quote == "USD"
    assert p.name == "EURUSD"


def test_pair_rejects_same_currency() -> None:
    with pytest.raises(ValueError):
        FxPair(base="USD", quote="USD")


def test_pair_rejects_lowercase() -> None:
    with pytest.raises(ValueError):
        FxPair(base="eur", quote="USD")


def test_pair_parse_too_short() -> None:
    with pytest.raises(ValueError):
        FxPair.parse("EU")


def test_rate_valid() -> None:
    r = FxRate("EURUSD", datetime(2024, 1, 1, tzinfo=UTC), 1.10)
    assert r.rate == 1.10


def test_rate_rejects_naive_date() -> None:
    with pytest.raises(ValueError):
        FxRate("EURUSD", datetime(2024, 1, 1), 1.10)


def test_rate_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        FxRate("EURUSD", datetime(2024, 1, 1, tzinfo=UTC), 0.0)
