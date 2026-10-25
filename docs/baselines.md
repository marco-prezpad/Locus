# Baselines

Reference implementations of naive trading strategies, plus the
vectorized evaluator that turns weights into an equity curve. Used to
answer "does any signal survive net of costs?" (G0 at F7).

## 1. What is a baseline

A baseline is a **reference strategy** with no learned parameters. If a
specialized model (F9) or ML baseline (F5.2) cannot beat the naive
baselines, the model is not adding value.

Four baselines in F5.1:

| Strategy | Weights |
| :--- | :--- |
| `BuyAndHoldBTC` | 1.0 on BTC, 0 elsewhere |
| `EqualWeightUniverse` | 1/N on each of the N available assets |
| `MomentumTopK(k)` | 1/K on the top K by `return_20` |
| `RandomTopK(k, seed)` | 1/K on a random K (seeded) |

ML baselines (Linear, LightGBM) are deferred to F9, alongside the
specialized models, to avoid duplicating the fit/predict pipeline.

## 2. Strategy interface

```python
class BaselineStrategy(Protocol):
    @property
    def name(self) -> str: ...
    def generate_weights(self, df: pl.DataFrame, rebalance_freq: int) -> pl.DataFrame: ...
```

generate_weights returns DataFrame[timestamp, asset_id, weight]
with one row per (rebalance_timestamp, asset). Weights sum to <= 1 at
each timestamp; the difference is cash, which earns 0.

Rebalance timestamps are every rebalance_freq-th unique timestamp,
starting from the first timestamp where all required features are
available.

## 3. One-bar-delay invariant

The evaluator respects the SignalExecutionContract:

Features at t → weights at t (using close of t).

Execution at open of t+1 (the next bar's open).

Holding until open of t+1+h, where h = holding_period_bars.

Return per asset: open(t+1+h) / open(t+1) - 1.

Portfolio return: sum_i w_i * asset_return_i (cash contributes 0).

This matches the plan §5.7 one-bar-delay. Executing at close of t
would be look-ahead; executing at open of t+1 is the earliest honest
point.

## 4. The vectorized evaluator (F5.1) vs the event-driven backtester (F6)

F5.1 (this layer) is a simplified evaluator:

No fees, spread, slippage, or market impact.

Fixed holding period per rebalance.

Continuous position (no execution failure, no partial fills).

Assumes aligned bars across assets.

Its purpose is comparison, not realism. It answers: "on average,
does this strategy have a positive expected return?" without modeling
execution costs.

F6 adds the event-driven backtester with:

Fee model (maker/taker).

Spread model.

Slippage model.

Market impact model (sqrt_volume).

One-bar-delay enforced by the clock (not by shift math).

Capital-aware sizing.

Three cost scenarios (optimistic, central, pessimistic).

G0 uses F6's results, not F5.1's. F5.1 is a fast loop to iterate
on strategies; F6 is the honest cost of the best one.

## 5. Metrics

| Metric | Formula |
|---|---|
| total_return | equity[-1] - 1.0 |
| cagr | equity[-1]^(1/years) - 1.0 |
| volatility | std(returns, ddof=1) * sqrt(periods_per_year) |
| sharpe | mean(excess) / std(excess, ddof=1) * sqrt(periods_per_year) |
| sortino | mean(excess) / std(downside, ddof=1) * sqrt(periods_per_year) |
| max_drawdown | min(equity / cummax(equity) - 1) |
| calmar | cagr / \|max_drawdown\| |
| hit_rate | mean(returns > 0) |
| turnover | mean over r of sum_i \|w[r,i] - w[r-1,i]\| / 2 |

periods_per_year depends on rebalance frequency: 365 for daily, 52
for weekly, 8760 for hourly (if the strategy rebalances every bar).

risk_free_rate_annual is subtracted per period (rf / periods_per_year)
in Sharpe and Sortino.

## 6. Interpreting results

Look for both:

Positive Sharpe (risk-adjusted return > 0): minimum bar.

CAGR > 0: absolute return positive.

Look at relative to BuyAndHoldBTC:

If MomentumTopK Sharpe > BuyAndHoldBTC Sharpe by a meaningful
margin, there is weak evidence of signal.

If MomentumTopK ≫ RandomTopK, the signal is not pure noise (random
also picks from the universe but has no feature).

Red flags in F5.1:

Sharpe < 0 for all strategies: the market is bearish, nothing helps.

RandomTopK beats MomentumTopK: the "signal" is noise.

MomentumTopK beats BuyAndHold by a huge margin: suspicious, check
for leakage in F6.

## 7. Limitations

No costs: F5.1 returns are gross. The plan's G0 gate uses F6's
net-of-cost results.

Aligned data required: if an asset is missing at a timestamp, its
weight contributes 0 to the portfolio return. For clean crypto data
this is rare; for equities with gaps and corporate actions it would
need careful handling in F6.

No position limits: a strategy could go 100% on one asset. Position
limits are a risk-management concern (F13), not an evaluator concern.

No slippage on rebalance: assumes we can buy/sell at the exact
open price. F6 adds slippage and impact.

Single-currency: all PnL is in the settlement currency (USDT for
crypto). Multi-currency handling is documented in currency.md.

## 8. FundingLongShortTopK (B5)

Market-neutral funding strategy: long the assets with the lowest funding
z-score, short the assets with the highest, dollar-neutral by construction.

### Signal

At each rebalance timestamp `t`:

1. Filter to assets with non-null `funding_zscore_30d` at `t` (warm-up
   excluded). Nulls propagate per-asset; a timestamp with < 2 valid assets
   generates no weights.
2. Rank ascending by the feature: lowest funding first.
3. Long = top `k` (lowest funding), short = bottom `k` (highest).
   If `2*k > n_valid`, `k_eff = n_valid // 2`.
4. Weights: `+gross/(2*k_eff)` for each long, `-gross/(2*k_eff)` for each
   short.
5. Dollar-neutral: `sum(w) = 0`, `sum(|w|) = gross_exposure`.

Execution: one-bar-delay (`VectorizedEvaluator` opens at `t+1 open`, holds
for `holding_period_bars`, closes at `t+1+h` open). No look-ahead: features
at `t` only use data with timestamp <= `t`.

### Parameters

| Parameter | Default | Scale |
|---|---|---|
| `feature` | `funding_zscore_30d` | Any funding feature from B2 |
| `k` | `2` | 2-5 for ~10 assets, 10-20 for 200+ assets |
| `gross_exposure` | `1.0` | 1.0 conservative, 2.0 long/short equity standard |
| `horizon_bars` | `8` | Must be <= `rebalance_bars` |
| `rebalance_bars` | `8` | Must be >= `horizon_bars` |

`k` and `gross_exposure` are intended to be tuned from config/CLI as the
universe grows; no code changes needed.

### Interaction with funding cash flows

The strategy itself does not model funding — that is the evaluator's job
(B4). Sign convention: positive funding rate ⇒ longs pay shorts.
Because the strategy is short high-funding assets and long low-funding
assets, the expected total funding cost should be <= 0 (the shorts receive
or pay little). `comparison_table()` exposes `funding_cost` for verification.

### Validation

- `horizon_bars <= rebalance_bars` enforced at construction. With
  `(8, 8)` no labels overlap and no uniqueness weights are required.
  With `rebalance < horizon`, the constructor raises a `ValueError`.
- Leakage test: `test_longshort_no_leakage_under_truncation` pins
  invariance under truncation (same pattern as B2's funding leakage test).

## 9. Cross-references

- `docs/features.md` — source of return_20 and other features.
- `docs/datasets.md` — labels, splits, n_eff.
- `docs/data.md` — where prices come from.
- `src/locus/baselines/` — implementation.
- `src/locus/evaluation/metrics.py` — metrics (shared with F6).
