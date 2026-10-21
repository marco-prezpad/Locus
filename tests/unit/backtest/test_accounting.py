"""
test_accounting.py

Cash and positions accounting.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

import pytest

from locus.backtest.accounting import PortfolioAccounting


def test_initial_state() -> None:
    a = PortfolioAccounting(cash=1000.0)
    assert a.cash == 1000.0
    assert a.positions == {}


def test_buy_updates_cash_and_position() -> None:
    a = PortfolioAccounting(cash=1000.0)
    a.apply_fill("BTC", 0.001, 30000.0)
    assert a.cash == pytest.approx(970.0)
    assert a.get_position("BTC") == pytest.approx(0.001)


def test_sell_updates_cash_and_position() -> None:
    a = PortfolioAccounting(cash=0.0)
    a.apply_fill("BTC", 0.001, 30000.0)  # buy
    a.apply_fill("BTC", -0.001, 31000.0)  # sell
    assert a.get_position("BTC") == 0.0
    assert "BTC" not in a.positions


def test_equity_marks_to_market() -> None:
    # Start with 30 USDT, buy 0.001 BTC at 30000 → cash = 0, pos = 0.001
    a = PortfolioAccounting(cash=30.0)
    a.apply_fill("BTC", 0.001, 30000.0)
    # Mark at buy price → equity unchanged
    assert a.equity({"BTC": 30000.0}) == pytest.approx(30.0)
    # Mark at higher price → equity increases by position P&L
    assert a.equity({"BTC": 31000.0}) == pytest.approx(31.0)


def test_equity_with_borrowed_cash() -> None:
    """Buying without cash makes cash negative; equity is still
    mark-to-market correct (P&L = position P&L)."""
    a = PortfolioAccounting(cash=0.0)
    a.apply_fill("BTC", 0.001, 30000.0)
    # -30 cash + 0.001 BTC × 30000 = 0
    assert a.equity({"BTC": 30000.0}) == pytest.approx(0.0)
    # -30 cash + 0.001 BTC × 31000 = 1.0 (P&L = +1 USDT)
    assert a.equity({"BTC": 31000.0}) == pytest.approx(1.0)


def test_equity_ignores_unknown_assets() -> None:
    a = PortfolioAccounting(cash=100.0)
    a.apply_fill("BTC", 0.001, 30000.0)
    eq = a.equity({})
    assert eq == pytest.approx(70.0)  # cash only
