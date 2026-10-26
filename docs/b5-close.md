# B5 (pivote B) — Closure note

**Code:** PB-20261025-01
**Hypothesis family:** funding long/short top-K, market-neutral
**Plan version:** v3.1
**Closure date:** 2026-10-26
**Status:** CLOSED — verdict NEGATIVE

---

## 1. Hypothesis

Over universe U10 (10 liquid Binance Futures perpetuals), a market-neutral
portfolio long low-funding assets and short high-funding assets, rebalanced
at a 168h horizon, achieves a net-of-costs Sharpe above the baselines
(B&H BTC, equal weight) in a statistically significant and regime-consistent
way.

## 2. Configuration evaluated

- **Universe:** U10 — BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, DOT, LINK
  - 170,200 H1 bars
  - 21,270 funding events (Binance Futures, 10 assets)
  - Common window: 2024-10-21 20:00 UTC → 2026-09-30 23:00 UTC
- **Features:** funding_zscore_30d, funding_zscore_7d, funding_xs_rank,
  funding_xs_zscore, funding_trend_7d
- **Strategy:** `FundingLongShortTopK`, dollar-neutral, gross=1.0
- **k:** 1, 2, 3, 4, 5
- **Horizon / rebalance:** 168h (primary), 24h (exploratory)
- **Costs:** central (fee 10, spread 5, slippage 5, impact 1 bps per side)
- **Walk-forward:** 3 folds, 34 periods per fold
- **Regime filter:** BTC 30d trend > 0 → cash (tested in B5.11)

## 3. Real metrics (walk-forward, event-driven engine)

### Primary feature: funding_zscore_30d, k=4

| Fold | Period | Sharpe | Return | MaxDD | Funding received |
|------|--------|--------|--------|-------|------------------|
| 0 | 2024-10-21 → 2025-06-15 | +0.554 | +7.85% | −22.6% | +28.49 USDT |
| 1 | 2025-06-15 → 2026-02-06 | −0.493 | −3.63% | −9.7% | +37.44 |
| 2 | 2026-02-06 → 2026-09-30 | −1.757 | −13.95% | −18.1% | +46.88 |
| **agg** | 102 returns | **−0.224** | **−10.56%** | −32.1% | +112.81 |

### Full trial sweep

| Trial | Fold 0 | Fold 1 | Fold 2 | Agg Sharpe | Return | Folds + |
|-------|--------|--------|--------|------------|--------|---------|
| zscore_30d, k=1, 168h | +0.82 | −0.36 | −1.26 | +0.137 | −5.5% | 1/3 |
| zscore_30d, k=2, 168h | +0.74 | −0.40 | −2.68 | −0.152 | −17.0% | 1/3 |
| zscore_30d, k=3, 168h | +0.21 | −1.49 | −2.28 | −0.644 | −27.1% | 1/3 |
| zscore_30d, k=4, 168h | +0.55 | −0.49 | −1.76 | −0.224 | −10.6% | 1/3 |
| zscore_30d, k=5, 168h | +0.70 | −0.06 | −2.19 | −0.081 | −5.8% | 1/3 |
| zscore_7d, k=4, 168h | +0.59 | +0.38 | −2.21 | −0.182 | −8.2% | 2/3 |
| trend_7d, k=4, 168h | +2.01 | −1.64 | −2.30 | +0.217 | +5.0% | 1/3 |
| zscore_30d, k=4, BTC filter | +1.01 | −1.95 | −1.54 | −0.094 | −4.5% | 1/3 |

**No trial meets E1** (Sharpe > best baseline + 0.2; B&H BTC walk-forward
Sharpe +0.344, threshold +0.544). **No trial meets E3** (≥70% positive
folds). **Fold 2 is negative in every trial.**

## 4. Controls C1-C3

Not formally executed. This note **does not claim significance**; the
verdict is NEGATIVE because E1 and E3 are not met, not because controls
failed.

## 5. Verdict (per pre-registered criteria)

**NEGATIVE.** No trial meets E1 (net Sharpe > baseline + 0.2) nor E3 (≥70%
positive folds). Aggregate return is negative in 7 of 8 trials. The only
positive-return trial (trend_7d, +5.0%) has fold 0 +2.01 and fold 2 −2.30:
unstable, not robust.

## 6. Methodological corrections (retrospective)

1. **Trials undercounted.** The DSR bootstrap used n_trials=1 or 5. The
   real number of configurations tested in B5 exceeds 40. All p-values
   published during exploration (0.37, 0.60, 0.94) are overoptimistic.
   Action: P0 reconstructs the registry.
2. **Costs underestimated in vectorized.** The 20 bps × turnover proxy
   underestimated the real engine cost (~8.5% per fold at 168h). B5
   vectorized results **are not cited as evidence**.
3. **Verdicts reinterpreted.** During exploration the same result was
   described as NEGATIVE, INCONCLUSIVE, and POSITIVE as diagnostics were
   added. Rule: the verdict is fixed by the pre-registered criteria and
   is not reinterpreted after seeing the data. **This note applies that
   rule:** the verdict is NEGATIVE because E1/E3 are not met,
   independently of whether later diagnostics suggest regime or
   dispersion effects.
4. **Dispersion threshold calibrated in-sample.** The +2.05 Sharpe in
   high-dispersion and −0.62 in low-dispersion were computed on the same
   sample used to define the threshold. This cannot be used as a rule or
   as evidence of edge. Documented as an exploratory observation only.

## 7. Complementary diagnostics (do not change the verdict)

- **BTC regime:** Sharpe +0.57 in BTC-down, ~0.00 in BTC-up. Not
  market-neutral in practice, but neither regime shows a robust edge
  (one is ~0, the other has −25% MaxDD).
- **Cross-sectional dispersion:** calibrated in-sample, discarded.
- **Rolling windows:** 71% positive but magnitudes erratic (Sharpe
  between −5.3 and +7.2). Sign consistency without magnitude consistency
  is not edge.
- **Concentration:** AVAX accounts for 76% of gross return. Would violate
  E6.

## 8. Lessons (plan §6.4)

- A favorable vectorized result is not evidence of edge. The gap with the
  cost-aware engine exceeded 50% of the return.
- At short horizons in crypto, execution cost outweighs any signal
  improvement.
- An uncounted trial is a false positive waiting to happen.
- A verdict reinterpreted after seeing results is not a verdict.

## 9. Known limitations

- **Survivorship bias (U10):** the universe was chosen with 2026
  information. Any conclusion in this note is subject to R6. Repeat with
  the reconstructed universe (P2) before citing as definitive evidence.
- **Short history (~2 years):** 101-102 periods per trial at 168h. Sharpe
  95% CI is wide (±0.2). R3.
- **Engine not formally validated:** P6 is operational but formal
  validation (consistency tests, slippage calibrated on real trades)
  remains for P0.
- **Trial registry incomplete:** reconstructed in P0 deliverable 2.

## 10. Next step per plan

Rama C (diagnosis) of plan v3.1, §13:
- Applicable diagnosis: **D2 (signal without net edge)** and **D3
  (unstable signal)**.
- D2 → A1 (edge threshold) or A2 (alternative horizon).
- D3 → reopened only if the regime was known before looking at the data.
  It was not. Treated as D1 for this variable.
- **Before A1/A2: P0 (consolidation).** Plan §7 is explicit: *"The next
  step is to consolidate the platform (P0) before testing new
  hypotheses."*
