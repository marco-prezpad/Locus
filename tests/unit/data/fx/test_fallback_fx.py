"""
test_fallback_fx.py

FallbackFxProvider picks the first non-empty provider.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from locus.data.fx.base import FX_PROVIDER_SCHEMA
from locus.data.fx.fallback_fx import FallbackFxProvider
from locus.data.fx.types import FxProviderError


def _df(n: int = 1) -> pl.DataFrame:
    rows = [
        {
            "pair": "USDTUSD",
            "date": datetime(2024, 1, 1 + i, tzinfo=UTC),
            "rate": 0.999 + 0.0001 * i,
        }
        for i in range(n)
    ]
    return pl.DataFrame(rows, schema=FX_PROVIDER_SCHEMA)


class _Success:
    source = "success"

    def fetch_rates(self, pair, start, end):
        return _df(2)


class _Fail:
    source = "fail"

    def fetch_rates(self, pair, start, end):
        raise FxProviderError("boom")


class _Empty:
    source = "empty"

    def fetch_rates(self, pair, start, end):
        return pl.DataFrame(schema=FX_PROVIDER_SCHEMA)


def _range() -> tuple[datetime, datetime]:
    return (
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 5, tzinfo=UTC),
    )


def test_first_wins() -> None:
    chain = FallbackFxProvider([_Success(), _Fail()])
    df = chain.fetch_rates("USDTUSD", *_range())
    assert df.height == 2


def test_second_used_on_failure() -> None:
    chain = FallbackFxProvider([_Fail(), _Success()])
    df = chain.fetch_rates("USDTUSD", *_range())
    assert df.height == 2


def test_second_used_on_empty() -> None:
    chain = FallbackFxProvider([_Empty(), _Success()])
    df = chain.fetch_rates("USDTUSD", *_range())
    assert df.height == 2


def test_all_fail_raises() -> None:
    chain = FallbackFxProvider([_Fail(), _Fail()])
    with pytest.raises(FxProviderError):
        chain.fetch_rates("USDTUSD", *_range())


def test_all_empty_returns_empty() -> None:
    chain = FallbackFxProvider([_Empty(), _Empty()])
    df = chain.fetch_rates("USDTUSD", *_range())
    assert df.height == 0


def test_empty_provider_list_raises() -> None:
    with pytest.raises(ValueError):
        FallbackFxProvider([])
