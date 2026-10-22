"""
engine.py

BacktestEngine: drives the loop from weights DataFrame + OHLCV to a
BacktestReport.

Timeline:
- rebalance_ts[0]: strategy generates weights from close of t0.
- t0+1 open: broker executes.
- Hold until close of rebalance_ts[1].
- rebalance_ts[1]: new weights from close of t1.
- t1+1 open: broker executes.
- ...

equity_curve:
- [0] = initial capital
- [i] = equity at close of rebalance_ts[i] (i = 1..N-1)
- [N] = equity at close of final bar

returns[i] = equity_curve[i+1] / equity_curve[i] - 1.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from locus.backtest.accounting import MissingPriceError, PortfolioAccounting
from locus.backtest.broker import SimulatedBroker
from locus.backtest.config import BacktestConfig
from locus.backtest.events import BacktestReport, FillEvent
from locus.backtest.market_state import enrich_with_market_state
from locus.evaluation.metrics import compute_metrics
from locus.infrastructure.logging import LoggerFactory


class BacktestEngine:
    def __init__(self, config: BacktestConfig, broker: SimulatedBroker) -> None:
        self._config = config
        self._broker = broker
        self._log = LoggerFactory.get("backtest.engine")

    def run(self, strategy_weights: pl.DataFrame, ohlcv: pl.DataFrame) -> BacktestReport:
        ohlcv_ext = enrich_with_market_state(ohlcv, bars_per_day=self._config.bars_per_day).sort(
            ["asset_id", "timestamp"]
        )

        timestamps = sorted(ohlcv_ext["timestamp"].unique().to_list())
        ts_index = {ts: i for i, ts in enumerate(timestamps)}

        lookup: dict[str, dict[datetime, dict]] = {}
        for row in ohlcv_ext.iter_rows(named=True):
            lookup.setdefault(row["asset_id"], {})[row["timestamp"]] = {
                "open": row["open"],
                "close": row["close"],
                "sigma": row["sigma_daily"] or 0.0,
                "volume": row["daily_volume_quote"] or 0.0,
            }
        assets = list(lookup.keys())

        def prices_at(ts: datetime, field: str) -> dict[str, float]:
            return {a: lookup[a][ts][field] for a in assets if ts in lookup[a]}

        def next_ts(ts: datetime) -> datetime | None:
            i = ts_index.get(ts)
            if i is None or i + 1 >= len(timestamps):
                return None
            return timestamps[i + 1]

        rebalance_ts = sorted(strategy_weights["timestamp"].unique().to_list())
        if not rebalance_ts:
            return self._empty_report()

        accounting = PortfolioAccounting(cash=self._config.capital, positions={})
        equity_curve = [self._config.capital]
        all_fills: list[FillEvent] = []
        n_orders_generated = 0

        for i, ts in enumerate(rebalance_ts):
            weights_df = strategy_weights.filter(pl.col("timestamp") == ts)
            target_weights = {
                row["asset_id"]: float(row["weight"]) for row in weights_df.iter_rows(named=True)
            }
            n_orders_generated += len(target_weights)

            close_prices = prices_at(ts, "close")
            current_equity = accounting.equity(close_prices)

            exec_ts = next_ts(ts)
            if exec_ts is None:
                continue

            open_prices = prices_at(exec_ts, "open")
            sigma = {a: lookup[a][ts]["sigma"] for a in assets if ts in lookup[a]}
            volume = {a: lookup[a][ts]["volume"] for a in assets if ts in lookup[a]}

            fills = self._broker.execute_rebalance(
                timestamp=exec_ts,
                current_equity=current_equity,
                target_weights=target_weights,
                open_prices=open_prices,
                sigma_daily=sigma,
                daily_volume_quote=volume,
                accounting=accounting,
            )
            all_fills.extend(fills)

            mark_ts = rebalance_ts[i + 1] if i + 1 < len(rebalance_ts) else timestamps[-1]
            mark_prices = prices_at(mark_ts, "close")
            try:
                mark_equity = accounting.equity(mark_prices)
            except MissingPriceError as exc:
                raise MissingPriceError(
                    f"backtest {self._config.strategy_name!r} "
                    f"failed at {mark_ts}: "
                    f"{exc}. This usually means the OHLCV has a gap "
                    f"at a rebalance timestamp. Run "
                    f"validate_ohlcv.check_continuity before the backtest."
                ) from exc
            equity_curve.append(mark_equity)

        equity = np.array(equity_curve, dtype=np.float64)
        returns = (
            equity[1:] / equity[:-1] - 1.0 if len(equity) > 1 else np.array([], dtype=np.float64)
        )
        metrics = compute_metrics(
            returns, self._config.periods_per_year, self._config.risk_free_rate_annual
        )

        total_cost_quote = float(sum(f.cost_quote for f in all_fills))
        total_cost_bps = (
            (total_cost_quote / self._config.capital) * 1e4 if self._config.capital > 0 else 0.0
        )

        report = BacktestReport(
            strategy_name=self._config.strategy_name,
            capital=self._config.capital,
            equity_curve=equity,
            returns=returns,
            metrics=metrics,
            fills=all_fills,
            n_orders_generated=n_orders_generated,
            n_orders_executed=len(all_fills),
            n_orders_skipped_min=self._broker.n_skipped_min,
            n_orders_skipped_lot=self._broker.n_skipped_lot,
            total_cost_quote=total_cost_quote,
            total_cost_bps=total_cost_bps,
        )
        self._log.info(
            "backtest_completed",
            extra={
                "strategy": self._config.strategy_name,
                "capital": self._config.capital,
                "n_periods": int(metrics.n_periods),
                "sharpe": round(float(metrics.sharpe), 3),
                "total_return": round(float(metrics.total_return), 4),
                "executed": report.n_orders_executed,
                "skipped_min": report.n_orders_skipped_min,
                "cost_bps": round(report.total_cost_bps, 2),
            },
        )
        return report

    def _empty_report(self) -> BacktestReport:
        return BacktestReport(
            strategy_name=self._config.strategy_name,
            capital=self._config.capital,
            equity_curve=np.array([self._config.capital]),
            returns=np.array([]),
            metrics=compute_metrics(np.array([]), self._config.periods_per_year),
        )
