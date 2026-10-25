# Backtester

Event-driven backtester with realistic costs, capital-aware execution,
and multi-scenario evaluation. Answers: **does the strategy survive
fees, spread, slippage, market impact, and the capital constraints of
the user's account?**

## 1. Why F6 and not F5.1

F5.1's `VectorizedEvaluator` produces **gross** returns: no fees, no
spread, no slippage, no impact, no min order size. That is fine to
compare strategies under idealised conditions, but it systematically
overstates Sharpe for high-turnover strategies and does not answer the
question that matters: does the strategy survive reality?

F6 adds:

- Fee, spread, slippage, market impact (per trade, from `costs.py`).
- Min order notional (skip below, per scenario).
- Lot size rounding (per asset).
- One-bar-delay execution, enforced by the engine.
- Three cost scenarios: optimistic, central, pessimistic.
- Seven capital scenarios: $1, $10, $100, $1k, $10k, $100k, $1M.

The output is a `BacktestReport` per (strategy, capital, cost scenario)
with net Sharpe, execution rate, and cost breakdown.

## 2. Execution model

**Timeline:**
```text
t0 close: strategy generates weights w_0
t0+1 open: broker executes rebalance at open prices
t0+1..t1: positions held
t1 close: equity marked at close
t1 close: strategy generates weights w_1
t1+1 open: broker executes rebalance
```

**Per-asset execution at rebalance:**

1. `target_notional = current_equity × weight`
2. `target_qty = floor(target_notional / open_price / lot_size) × lot_size`
3. `delta_qty = target_qty - current_qty`
4. If `|delta_qty × open_price| < min_order_notional` → **skip**.
5. `cost_bps = fee + half_spread + slippage + impact`.
6. `fill_price = open_price × (1 ± cost_bps/1e4)` (buy: +, sell: -).
7. Update cash and position.

The cost is embedded in the fill price, so cost accounting is automatic:
`cash -= delta_qty × fill_price`.

## 3. Capital scenarios

| Scenario | Capital | Min order | Notes |
| :--- | :--- | :--- | :--- |
| `1` | $1 | $5 | Everything skipped; strategy = cash |
| `10` | $10 | $5 | 1-2 assets, marginal |
| `100` | $100 | $5 | Diversification limited |
| `1k` | $1,000 | $5 | Representative |
| `10k` | $10,000 | $5 | Representative, impact starts |
| `100k` | $100,000 | $5 | Impact meaningful |
| `1m` | $1,000,000 | $5 | Impact dominates |

The `CapitalScalingExperiment` runs all seven for a given strategy and
cost scenario. The comparison table shows execution rate (executed /
generated) and net Sharpe.

## 4. Cost scenarios

| Scenario | Fee | Spread | Slippage | k |
| :--- | :--- | :--- | :--- | :--- |
| optimistic | 5.0 | 2.0 | 2.0 | 0.5 |
| central | 10.0 | 5.0 | 5.0 | 1.0 |
| pessimistic | 20.0 | 10.0 | 15.0 | 1.5 |

Report **all three** in any final claim. A strategy that only survives
under optimistic is not a strategy.

## 5. Report fields

`BacktestReport`:

- `equity_curve`: array of length n_rebalances + 1.
- `returns`: per-rebalance net returns.
- `metrics`: Sharpe, Sortino, MDD, Calmar, hit_rate, turnover, etc.
- `fills`: list of executed trades with cost per trade.
- `n_orders_generated`: number of (asset, rebalance) pairs.
- `n_orders_executed`: number of fills.
- `n_orders_skipped_min`: skipped below min notional.
- `total_cost_quote`: total cost paid in quote currency.
- `total_cost_bps`: total cost as fraction of initial capital.

`execution_rate = executed / generated`. At $1, it is 0. At $1k+, it
is ~1.

## 6. Known limitations

- **Lot sizes are hardcoded** per asset in `config.py` (`_DEFAULT_LOT_SIZES`).
  They can be overridden per `BacktestConfig`. F4's multi-exchange work
  will source them from the exchange.
- **Min notional is per scenario**, not per asset. In reality it varies
  (Binance has $5 min for most pairs, $10 for some). F4 will add
  per-asset overrides.
- **No partial fills**: if the target quantity exceeds available
  liquidity, we still fill it. For small capital this is fine; for
  large capital the impact model accounts for part of the cost, but
  not the "cannot fill" case.
- **No funding rates**: spot only. Perpetuals add funding in B5.
- **No delisted assets in the universe**: prices always available.
- **Cash earns zero**: no money market, no USDT staking. Realistic for
  a CEX account with no yield on idle USDT.
- **No slippage of market impact on the exit**: the impact model is
  applied per trade symmetrically (buy and sell), but there is no
  "market moves against you because you sold" beyond the half-spread
  and impact already applied.

## 7. How to interpret results

**Compare net vs gross.** F5.1 gives gross; F6 gives net. A strategy
whose Sharpe goes from 0.5 to -0.3 has been destroyed by costs. A
strategy whose Sharpe goes from 0.5 to 0.4 is robust.

**Compare across capitals.** `MomentumTop2` at $10 does nothing. At
$10k it does everything. The Sharpe at $1k+ is the honest number.

**Compare across scenarios.** If Sharpe is positive under optimistic
only, it is not edge.

**Look at execution_rate.** At $10, momentum generates 4 orders per
rebalance (2 assets × 2 sides), and executes 0-1. The "portfolio" is
90%+ cash. That is the real answer for a user starting small.

## 8. Funding cash flows (B3)

The engine can apply perpetual funding as a separate cash flow, independent
of fills and execution costs.

- **Sign convention:** positive funding rate ⇒ longs pay shorts.
  `cash_flow = -qty * price * funding_rate`.
- **Timing:** funding events are applied at their own timestamp, using the
  `close` of the funding bar as reference price. No look-ahead: only events
  with `timestamp <= mark_ts` are applied before marking equity.
- **Isolation:** funding does **not** affect fill prices or execution costs.
  `test_funding_cash_flow_does_not_affect_fill_price` enforces this.
- **Report fields:** `BacktestReport.total_funding_quote` (signed) and
  `n_funding_events_applied`.
- **Disabling:** set `BacktestConfig.apply_funding=False` to ignore any
  funding DataFrame passed to `run`.
- **Attribution rule:** funding at timestamp `f` is attributed to the position
  that existed at `f`:
  - If `f < exec_ts` (the funding event precedes the fill of the current
    rebalance cycle), the **pre-fill** position is used (typically 0 for the
    first cycle, non-zero for subsequent ones).
  - Otherwise, the **post-fill** position is used.
  - Events that fall in `(prev_mark_ts, exec_ts)` are therefore attributed to
    the position held before the fill, not after. This prevents a buy executed
    at `exec_ts` from retroactively "owing" funding for an event that occurred
    hours earlier.

## 9. Cross-references

- `docs/baselines.md` — the strategies being evaluated.
- `docs/features.md` — the source features.
- `docs/datasets.md` — labels, splits, `n_eff`.
- `src/locus/backtest/` — implementation.
- Plan §13 — capital, budget, capacity.
