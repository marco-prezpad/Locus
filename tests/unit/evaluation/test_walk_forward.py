"""
test_walk_forward.py

WalkForwardBaselineEvaluator: window splitting, return concatenation,
aggregate metrics.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from locus.backtest.config import load_backtest_config
from locus.baselines import BuyAndHoldBTC
from locus.data.providers.base import BAR_SCHEMA
from locus.evaluation.walk_forward import WalkForwardBaselineEvaluator


def _df(n: int = 500) -> pl.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "asset_id": "BTC",
                "venue": "test",
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "timestamp": base + timedelta(hours=i),
                "open": 100.0 * (1.0001**i),
                "high": 100.0 * (1.0001**i) * 1.01,
                "low": 100.0 * (1.0001**i) * 0.99,
                "close": 100.0 * (1.0001**i),
                "volume": 1e6,
            }
            for i in range(n)
        ],
        schema=BAR_SCHEMA,
    )


def _cfg():
    return load_backtest_config(strategy_name="x", capital_name="10k", rebalance_freq_bars=24)


def test_evaluator_produces_n_folds() -> None:
    evaluator = WalkForwardBaselineEvaluator(_cfg(), n_folds=3)
    result = evaluator.run(BuyAndHoldBTC(), _df(500), rebalance_freq_bars=24)
    assert len(result.per_fold) == 3


def test_evaluator_concatenates_returns() -> None:
    evaluator = WalkForwardBaselineEvaluator(_cfg(), n_folds=3)
    result = evaluator.run(BuyAndHoldBTC(), _df(500), rebalance_freq_bars=24)
    # Each fold has non-empty returns
    fold_lengths = [f.n_returns for f in result.per_fold]
    assert sum(fold_lengths) == len(result.returns)


def test_evaluator_aggregate_metrics() -> None:
    evaluator = WalkForwardBaselineEvaluator(_cfg(), n_folds=2)
    result = evaluator.run(BuyAndHoldBTC(), _df(300), rebalance_freq_bars=24)
    assert result.metrics.n_periods == len(result.returns)
    assert result.strategy_name == "buy_and_hold_btc"


def test_evaluator_insufficient_data_raises() -> None:
    evaluator = WalkForwardBaselineEvaluator(_cfg(), n_folds=4)
    with pytest.raises(ValueError, match="not enough timestamps"):
        evaluator.run(BuyAndHoldBTC(), _df(5), rebalance_freq_bars=24)


def test_evaluator_rejects_single_fold() -> None:
    with pytest.raises(ValueError):
        WalkForwardBaselineEvaluator(_cfg(), n_folds=1)
