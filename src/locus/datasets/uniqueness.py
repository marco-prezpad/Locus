"""
uniqueness.py

Per-asset uniqueness weights for overlapping labels. Sample at t has
label interval [t, t+h); sample at t' overlaps if t' is inside that
window (half-open).

`weight(t) = 1 / concurrency(t)`, where `concurrency(t)` is the number
of samples whose label window contains t (per asset). For uniformly
spaced bars with no boundary effects, concurrency = h.

`n_eff = sum(weights)` approximates the number of independent
observations after accounting for overlap. For uniformly spaced bars
this is close to `n / h`.

Cross-asset correlation is not adjusted here; it is documented as a
post-MVP limitation.

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

import numpy as np
import polars as pl


def compute_uniqueness_weights(
    df: pl.DataFrame,
    horizon_bars: int,
) -> pl.DataFrame:
    """Add a `uniqueness_weight` column: 1/concurrency per asset.

    `concurrency(t)` counts the samples in the same asset whose label
    interval `[t', t'+h)` contains t. For uniformly spaced bars in the
    interior (no boundary effects), this is exactly `h`. The result is
    close to `n/h` when summed over a long series.
    """
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    if df.height == 0:
        return df.with_columns(pl.Series("uniqueness_weight", [], dtype=pl.Float64))

    result = df.sort(["asset_id", "timestamp"])
    # Datetime("us") cast to Int64 gives microseconds since epoch.
    # We only need the unit to be consistent within this function.
    ts_us = result["timestamp"].cast(pl.Int64).to_numpy()
    asset_ids = result["asset_id"].to_numpy()

    weights = np.zeros(len(result), dtype=np.float64)

    for asset in np.unique(asset_ids):
        idx = np.where(asset_ids == asset)[0]
        asset_ts = ts_us[idx]
        if len(asset_ts) < 2:
            weights[idx] = 1.0
            continue
        step = float(np.median(np.diff(asset_ts)))
        window = step * horizon_bars  # half-open [t, t + h*dt)
        # Samples whose label window contains this bar:
        # t' ∈ (asset_ts - window, asset_ts] → count via searchsorted.
        left = np.searchsorted(asset_ts, asset_ts - window, side="right")
        right = np.searchsorted(asset_ts, asset_ts, side="right")
        concurrency = np.maximum(right - left, 1).astype(np.float64)
        weights[idx] = 1.0 / concurrency

    return result.with_columns(pl.Series("uniqueness_weight", weights, dtype=pl.Float64))


def estimate_n_eff_simple(n_samples: int, horizon_bars: int) -> float:
    """First-order estimate of effective sample size: N / h. Use this
    as a sanity check against sum(uniqueness_weights)."""
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    return n_samples / horizon_bars
