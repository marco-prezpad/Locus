"""
market_state.py

Helpers to estimate per-bar market state needed by the cost models:
daily volatility (sigma_daily) and daily traded notional (V_daily).

These are estimated from OHLCV bars. Both are rolling, per asset, using
only data available at bar t (backward-looking).

- sigma_daily: std of 1-bar returns over `window` bars, scaled by
  `sqrt(bars_per_day)`. For 1h bars with `bars_per_day=24`, this gives
  the daily volatility as a fraction (e.g. 0.03 = 3%).
- daily_volume_quote: rolling mean of `volume * close` over `window`
  bars, multiplied by `bars_per_day`. This is the daily traded notional
  in quote currency, used as `V_daily` in the impact formula.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

import numpy as np
import polars as pl


def estimate_sigma_daily(
    df: pl.DataFrame,
    bars_per_day: int = 24,
    window: int = 24,
) -> pl.DataFrame:
    """Add `sigma_daily` column: rolling daily volatility per asset.

    Requires `close` column. Null for the first `window + 1` bars of
    each asset (returns need a shift, rolling needs window).
    """
    if bars_per_day <= 0:
        raise ValueError("bars_per_day must be positive")
    if window < 2:
        raise ValueError("window must be >= 2")
    result = df.sort(["asset_id", "timestamp"])
    returns = pl.col("close") / pl.col("close").shift(1).over("asset_id") - 1
    return result.with_columns(
        (returns.rolling_std(window).over("asset_id") * np.sqrt(bars_per_day)).alias("sigma_daily")
    )


def estimate_daily_volume_quote(
    df: pl.DataFrame,
    bars_per_day: int = 24,
    window: int = 24,
) -> pl.DataFrame:
    """Add `daily_volume_quote` column: rolling daily traded notional
    per asset, in quote currency.

    Requires `volume` and `close` columns. Null for the first `window-1`
    bars of each asset.
    """
    if bars_per_day <= 0:
        raise ValueError("bars_per_day must be positive")
    if window < 1:
        raise ValueError("window must be >= 1")
    result = df.sort(["asset_id", "timestamp"])
    notional = pl.col("volume") * pl.col("close")
    return result.with_columns(
        (notional.rolling_mean(window).over("asset_id") * float(bars_per_day)).alias(
            "daily_volume_quote"
        )
    )


def enrich_with_market_state(
    df: pl.DataFrame,
    bars_per_day: int = 24,
    volatility_window: int = 24,
    volume_window: int = 24,
) -> pl.DataFrame:
    """Add both `sigma_daily` and `daily_volume_quote` columns."""
    result = estimate_sigma_daily(df, bars_per_day, volatility_window)
    return estimate_daily_volume_quote(result, bars_per_day, volume_window)
