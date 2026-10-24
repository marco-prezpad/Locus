# G0 Protocol v3 — Funding Rates

**Pre-registered before any code is written.** Changes after this
document is committed require a new research cycle (new protocol
version), not an edit.

Author: Marco Pérez Padilla
Date: 2026-10-23
Pre-registration commit: <pending, will be filled when B.0 is committed>

---

## 1. Hypothesis

Funding rates of perpetual futures contain information about
positioning imbalance (crowding) that predicts short-horizon returns.

Mechanism: a perpetual futures contract pays funding every 8h. When
funding is extreme positive, longs are paying shorts to stay in the
trade — longs are crowded and over-leveraged. A mean-reverting
liquidation or de-leveraging is more likely, predicting negative
returns on the crowded side. Symmetric for extreme negative funding
(shorts crowded, positive returns expected).

The signal is **cross-sectional**: at each funding timestamp, rank
assets by funding rate or funding z-score, long the low (shorts
crowded), short the high (longs crowded).

---

## 2. Venue (fixed before code)

**Venue: `<VENUE>`** (decision from accessibility test, see §2.1).

### 2.1. Accessibility requirement

Before writing any provider, the developer must verify that the venue's
public API responds from Spain (target deployment region). Command:

```powershell
curl -s -o /dev/null -w "%{http_code}" "<venue funding endpoint>"
```

If HTTP 200, venue is accepted. If HTTP 451/403/timeout, the venue
is rejected and the next candidate in the list is tested:

- binance_futures (fapi.binance.com)
- bybit_perp (api.bybit.com)
- okx_perp (www.okx.com)
- Historical dumps from data.binance.vision (no live API)

All results in this protocol are conditioned on the chosen venue.
Cross-venue robustness is out of scope for this cycle.

---

## 3. Universe (point-in-time, no survivorship bias)

Universe: 5 perpetual futures on the chosen venue:
BTC, ETH, SOL, BNB, XRP (vs USDT, linear perps).

Point-in-time requirement (prerequisite, not follow-up):

Each perp has:

- `asset_id`: canonical (BTC, ETH, ...)
- `venue`: the venue from §2
- `list_date`: first funding timestamp available
- `delist_date`: last funding timestamp (or None if still live)

A perp is in the universe at timestamp t iff
`list_date <= t < delist_date`.

Delisted perps are excluded from the universe, never valued at 0.
Binance, Bybit, and OKX have all delisted and relisted BTC perps
(contract migrations, COIN-M vs USDT-M). The FundingStore partitions
by venue/asset/year and reads only timestamps in [list, delist).

Test: `test_universe_point_in_time` — synthesize a perp with
list_date=2023-01-01, delist_date=2024-06-01. Verify it appears in
the universe on 2023-06-01 and is absent on 2024-07-01, with no
value at 0 for the missing period.

---

## 4. Data requirements

### 4.1. Funding data

- Cadence: venue-native (8h for most perps; 4h/1h for some newer pairs).
- History: 3 years (2023-10-01 to 2026-10-01).
- Fields: `asset_id`, `venue`, `symbol`, `timestamp` (UTC), `funding_rate` (signed float).
- Storage: `data/processed/funding/venue={v}/asset={a}/year={y}/rates.parquet`.
- Raw layer: `data/raw/funding/{venue}/{asset}/raw.parquet` with the literal API response + `fetched_at` timestamp.

### 4.2. Price data

Source: existing OHLCV store (Binance spot, from F2).

Reason: spot prices for valuation, perp prices for funding
(avoid basis contamination in the signal).

Alternative if needed: perp OHLCV, if spot is unreliable for
the chosen venue.

### 4.3. Reproducibility

All scripts must accept `--end-date` with a fixed default.
Default: 2026-10-01. Pre-existing scripts (smoke_f4, smoke_f6,
g0_report, null_test_random) must be updated to accept this flag
before B.1 starts.

Test: `test_reproducibility` — run `g0_report --end-date 2026-10-01`
twice, verify identical output (byte-for-byte on the JSON, or
floating-point-identical on the metrics).

---

## 5. Features

All features are computed at the funding timestamp, using only data
with timestamp <= t (point-in-time discipline, see §6).

### 5.1. Per-asset

- `funding_rate`: raw signed funding rate.
- `funding_zscore_30d`: rolling z-score over 90 funding periods (30 days at 8h cadence).
- `funding_zscore_7d`: rolling z-score over 21 funding periods (7 days).
- `funding_extreme_flag`: binary, `|funding_zscore_30d| > 2`.
- `funding_trend_7d`: change in funding over 21 periods (7 days).

### 5.2. Cross-sectional

- `funding_xs_rank`: percentile rank of funding rate across assets at each timestamp, in (0, 1).
- `funding_xs_zscore`: cross-sectional z-score of funding rate.

### 5.3. Explicitly excluded (documented)

Price-based features (returns, momentum, MA distance). Funding
strategies must be tested without price features to isolate the
funding signal. If funding + price beats funding alone, that is a
separate hypothesis (multi-factor), tested in a different cycle.

On-chain, sentiment, macro. Out of scope.

---

## 6. Point-in-time discipline

### 6.1. Funding timestamp semantics

The exchange publishes the funding rate at `fundingTime` (e.g.,
00:00, 08:00, 16:00 UTC). The rate is known at that timestamp,
not before. It applies to the 8h period ending at that timestamp.

A feature at timestamp t may use funding data with
`funding_time <= t`. It may not use `funding_time > t`.

### 6.2. Alignment with price bars

Price bars are 1h, closing at HH:00 UTC. A funding event at 00:00
UTC is available for the price bar that closes at 00:00 UTC (same
timestamp), and for every subsequent bar. It is not available for
the 23:00 UTC bar, even though the funding period technically ends
at 00:00.

Rule: `price_bar_timestamp >= funding_timestamp` for the funding
to be usable.

### 6.3. Leakage test

`test_funding_no_lookahead`:

1. Build features on full funding history.
2. Truncate funding data at cutoff.
3. Recompute features.
4. Assert values at t <= cutoff are identical.

If a feature uses future funding (e.g., `shift(-1)`, forward-fill
from future), the test fails.

---

## 7. Strategies (market-neutral)

### 7.1. FundingLongShortTopK

At each funding timestamp:

- Rank assets by `funding_zscore_30d` (or another configured source).
- Long the k assets with the lowest funding (most negative z-score): shorts are crowded, likely to unwind upward.
- Short the k assets with the highest funding (most positive z-score): longs are crowded, likely to unwind downward.
- Equal-weight within each leg.
- Target gross exposure: 100% (50% long + 50% short).
- Target net exposure: 0% (market-neutral).

Legs may be empty if fewer than k assets have non-null funding
z-score at a timestamp. In that case, the strategy holds cash for the
empty leg.

### 7.2. Baselines (for comparison, not strategies under test)

- **BuyAndHoldBTC**: beta baseline.
- **MomentumTopK**: same horizon, same assets, price-based. Isolates funding from cross-sectional price momentum.
- **LongShortMomentumTopK**: same long/short structure but on price momentum. Isolates funding from the long/short mechanic.
- **RandomLongShortTopK**: random 2 long, 2 short at each rebalance. Null for the long/short mechanic (200 seeds, as in MVP-0).

### 7.3. Net exposure monitoring

At every rebalance, compute and log:

- Gross exposure (sum of absolute weights).
- Net exposure (sum of signed weights).
- Net BTC exposure (weight in BTC, signed).

If net BTC exposure > 0.20 at any rebalance, the strategy is not
market-neutral and the result is flagged as **beta-contaminated**. A
result that depends on net exposure does not count as funding edge.

---

## 8. Cost model (funding as signed cash flow)

### 8.1. Per-trade costs (existing)

`TransactionCostModel` (fee + spread + slippage + impact) applies to
each trade (entry and exit of every leg), same as in MVP-0. For perps:

- Fee: 5 bps maker / 10 bps taker (Binance Futures retail default).
- Spread: 2 bps.
- Slippage: 5 bps.
- Impact: sqrt law, k=1.0 central.

### 8.2. Funding flows (new)

Funding is not a fee. It is a signed cash flow.

At each funding timestamp, for each open position:

```text
funding_flow = -position_qty * notional_per_unit * funding_rate
```

- If long and `funding_rate > 0`: cash decreases (pay funding).
- If short and `funding_rate > 0`: cash increases (receive funding).
- If `funding_rate < 0`: signs invert.

The cash flow is applied at the exact funding timestamp, not
amortized. The BacktestEngine gains a "funding accrual" step that
runs before the regular rebalance loop.

Test: `test_funding_cash_flow_sign` — long position with positive
funding decreases cash; short position with positive funding increases
cash. Symmetric for negative funding.

### 8.3. Basis and roll

If the strategy holds the perp from entry to a funding timestamp, no
basis cost beyond the entry/exit fees. If it rolls the perp (closes
and reopens to avoid a delisting), the basis between old and new
contract is a real cost. For the 3-year test window, no delistings
are expected (see §3). Basis is documented as a known limitation
for longer windows.

### 8.4. Liquidation risk

Not modeled. A perp position with 50% gross exposure and daily
volatility of 5% requires ~150% margin to avoid liquidation under
adverse moves. The backtester does not model maintenance margin or
liquidations. Documented as a known limitation: real deployment must
size positions to survive liquidation (typically 2× maintenance
margin headroom).

---

## 9. Horizon and rebalance

- **Primary**: horizon 8h, rebalance 8h. The funding cadence and the signal cadence match. No label overlap (horizon == cadence → n_eff = N).
- **Secondary**: horizon 24h, rebalance 24h. Test robustness to a longer holding period.

Explicitly NOT tested in this cycle: horizon 168h (weekly, from
pivote A) or horizon 1h. Mixing horizons across cycles is what
invalidated pivote A's apparent signal.

---

## 10. Criterion (G0_CRITERION_V3)

```python
G0_CRITERION_V3 = G0Criterion(
    min_sharpe=0.0,
    min_dsr=0.80,                  # calibrated for 3 years of data
    available_years=3.0,
    max_pbo=0.50,
    benchmark_name="buy_and_hold_btc",
    n_trials_for_dsr=5,            # 5 funding feature combinations tested
    min_sharpe_vs_benchmark=0.15,  # higher than V2's 0.10
)
```

Rationale for `min_sharpe_vs_benchmark=0.15`: funding has a
stronger prior than technical momentum (economic mechanism + mixed
academic evidence). Requiring 0.15 above Buy&Hold rather than 0.10
is a stricter bar for a stronger claim.

---

## 11. Power analysis (computed before running)

With 3 years of funding data at 8h cadence:

- Total funding periods: ~3,285 (365 × 3 × 3 = 3,285).
- With 5 assets: ~16,425 asset-timestamps.
- Horizon 8h, rebalance 8h: no overlap. n_eff per asset-timestamp = 1.
- Effective sample size for the portfolio: ~3,285 (one portfolio observation per funding timestamp, aggregated across 5 assets).

Minimum detectable Sharpe (MDE), α=0.05 two-sided, power=0.80:

| n_eff | MDE (annualized) |
|------:|-----------------:|
| 3,285 | 0.07 |
| 1,000 | 0.13 |
| 500   | 0.18 |
| 250   | 0.26 |

With ~3,285 effective observations, the study can detect a Sharpe
of 0.10+ at 80% power. This is a much more powerful study than
pivote A (which had n_eff ≈ 547 with 5 years weekly). The minimum
detectable effect is well below the 0.15 margin required by
`min_sharpe_vs_benchmark`.

This is recorded before any data is fetched, to prevent
post-hoc rationalization.

---

## 12. Null tests

### 12.1. Random long/short

200 seeds of `RandomLongShortTopK(k=2)`. Null distribution of Sharpe.
Momentum-level comparison: funding strategy's percentile in the null.

### 12.2. Funding permutation

Permute funding timestamps within each asset (same shuffle applied
across assets to preserve cross-sectional structure). Recompute
strategy. 200 permutations. Null distribution of Sharpe under
"funding is temporally random".

Both nulls must be passed. A strategy that passes only one is
not considered a demonstrated edge.

---

## 13. Success criterion

A funding strategy passes G0-v3 if all of:

- DSR > 0.80.
- CI_low > 0.
- MinBTL <= 3.0 years.
- Sharpe > Buy&Hold Sharpe + 0.15.
- PBO < 0.50 across all strategies.
- p < 0.05 (one-sided) against both nulls (§12.1 and §12.2).
- Net BTC exposure <= 0.20 at every rebalance (market-neutral check).

---

## 14. Failure criterion

If no funding strategy passes all criteria in §13:

- Pivote B is negative. Documented in `governance.md` §7.d.
- No further pivots without a new hypothesis, new pre-registration, and a new cycle.

The project closes with: MVP-0 (technical features, negative), pivote
A (weekly horizon, negative), pivote B (funding rates, negative).
Three cycles, three negative gates, all documented.

---

## 15. What this protocol does NOT establish

Even if G0-v3 is positive, this protocol does not establish:

- That the edge is robust to cross-venue (only one venue tested).
- That the edge survives realistic slippage on larger capital (capital scenarios are limited to $10k in this cycle).
- That the edge survives market impact (impact model is simplified).
- That the edge exists in live trading (paper trading is a separate gate, not part of G0-v3).
- That delta-neutral (spot long + perp short) captures the same signal (a separate two-leg cost model is not implemented).

Even if G0-v3 is negative, this protocol does not establish:

- That funding rates carry no signal in other markets.
- That funding rates cannot be monetized with institutional cost structures.
- That other on-chain or macro features carry no signal.

---

## 16. Execution plan

| Phase | Deliverable | Duration |
|-------|-------------|----------|
| B.0 | This protocol committed | 1 day |
| B.1 | Data layer: funding provider, store, backfill 3y × 5 assets, raw+normalized | 1 week |
| B.2 | Features + leakage tests + point-in-time universe tests | 3 days |
| B.3 | Funding cash flow in backtester + regression tests | 2 days |
| B.4 | Long/short strategies + baselines + net exposure monitoring | 3 days |
| B.5 | G0-v3 report + both null tests | 3 days |
| B.6 | Analysis, documentation in `governance.md` §7.d, decision | 1 day |
| | **Total** | **~4 weeks** |

---

## 17. Cross-references

- Plan §8.15 — evaluation protocol.
- Plan §13 — capital and capacity.
- `docs/governance.md` §7.c — MVP-0 closure (previous cycle).
- `docs/significance.md` — DSR, PBO, MinBTL.
- `src/locus/evaluation/g0.py` — criteria V1, V2 (V3 added in B.5).
- `docs/data.md` — data layer (funding provider added in B.1).
