# Governance

The rules that keep the project honest. Pre-registered criteria,
experiment counting, one-shot holdout, pivot protocol. These rules
exist to prevent the failure mode that kills most quant projects:
testing hundreds of variants, picking the best, and calling it edge.

## 1. The three layers

| Layer | Purpose | Rules |
| :--- | :--- | :--- |
| **Exploration** | Iteration, notebooks, quick tests. | No constraints. Doesn't count toward `n_trials`. |
| **Selection** | Experiments that inform a decision (model, features, hyperparameters, thresholds). | Every run is recorded in `ExperimentRegistry` with `purpose=selection`. |
| **Final** | The one-shot evaluation against a pre-registered protocol. | Exactly one record with `purpose=final`. Cannot be re-run. |

## 2. Experiment counting

Every selection run adds to `n_trials`, the input to Deflated Sharpe
Ratio. The registry (`ExperimentRegistry`) is JSON-backed, atomic, and
ships with the repo. It is the source of truth for `n_trials`.

```text
data/processed/experiments/registry.json
```

`n_trials` = `count(selection) + count(final)`. Exploration is excluded
because it did not influence a decision.

**Honest count > flattering count.** If you tested 50 hyperparameters
during exploration and 5 made it to selection, count 5. If you used
exploration to *notice* which 5 to select, count 5, not 50. DSR asks
about the final trials, not every intermediate step.

## 3. Pre-registration (the protocol)

Before running the final evaluation, commit:

1. **`docs/g0_protocol.md`** with:
   - The exact criterion (`G0_CRITERION_V1` fields: min_sharpe,
     min_dsr, available_years, max_pbo, benchmark).
   - The exact dataset (universe, period, timeframes, cost scenario).
   - The exact strategies tested.
   - The exact test statistics (DSR, PBO, MinBTL, block bootstrap CI).
   - The commit hash of the code, data version, and feature version.

2. **The registry snapshot** (`n_trials` at the moment of
   pre-registration).

3. **The decision rule** (what outcome triggers positive vs negative
   vs inconclusive).

Then — and only then — run the final evaluation. **Any deviation after
the pre-registration is a protocol violation** and must be documented
as such.

## 4. The one-shot rule

The final evaluation runs **once**. If the outcome is positive, done.
If the outcome is negative, done — the negative result is documented
with the same rigor as a positive result.

**No second looks at the same dataset with the same criterion.**
`ExperimentRegistry.add(purpose="final", ...)` raises on the second
attempt. This is not a bug; it is the point.

If you want to test a different hypothesis (longer horizon, different
features, different universe), that is a **new research cycle** with
its own pre-registration, its own registry, and its own final
evaluation. The old negative result remains valid for the old setup.

## 5. Pivot protocol if G0 is negative

Per plan §8.15, a negative G0 is a valid outcome. The pivot options:

### 5.1. Pivot to a longer horizon

`future_return_24h` was chosen for the MVP. Test `future_return_1w`
or `future_return_1mo`. Longer horizons:
- Reduce turnover → less cost drag.
- Increase `n_eff` (fewer overlapping labels) → more statistical power.
- Match the historical evidence better (momentum in crypto works at
  weekly horizons, not daily).

### 5.2. Pivot to alternative features

The MVP uses 22 technical features (price, volume, volatility). Test:
- On-chain metrics (hash rate, active addresses, exchange flows).
- Funding rates.
- Sentiment (Twitter, news).
- Macro (DXY, rates, equities).
- Cross-asset (correlation with gold, S&P 500).

Each requires new data, new validation, and its own pre-registration.

### 5.3. Accept the MVP as a research platform

Declare that the MVP built:
- Full infrastructure (Docker, CI, 400+ tests, 94% coverage).
- Data layer (6 providers, universe history, FX).
- Feature engineering with leakage tests.
- Event-driven backtester with capital-aware costs.
- Significance framework (DSR, PBO, MinBTL, bootstrap).

And that the honest answer to "is there edge with this setup?" is
**no**.

**That is a successful MVP.** The platform is there. Anyone can test
a new hypothesis without rebuilding infrastructure.

## 6. What NOT to do

- Tune the criterion after seeing the numbers.
- Re-run the final evaluation "just to check".
- Count only "successful" experiments in `n_trials`.
- Cherry-pick the cost scenario (optimistic only).
- Cherry-pick the capital scenario (only $10k when smaller/larger
  fail).
- Report DSR without `n_trials`.
- Report Sharpe without CI.
- Report alpha without comparing to benchmark.

## 7. Reporting format

Every final report must include, for each strategy:

| Field | Required |
| :--- | :--- |
| Sharpe (annualized) | Yes |
| Total return | Yes |
| Max drawdown | Yes |
| **DSR** | Yes |
| **95% CI (bootstrap)** | Yes |
| **MinBTL** | Yes |
| **PBO (across strategies)** | Yes |
| **n_trials** | Yes |
| Benchmark comparison | Yes |
| **Decision** (positive / negative / inconclusive) | Yes |
| **Rationale** | Yes |

A report without any of these fields is not a final report.

## 7.b. Calibration of the criterion (v1 → v2)

The `G0_CRITERION_V1` was defined assuming the DSR 0.95 threshold is
achievable with the data available. Analysis at gate G0 (MVP-0,
2026-10-22) revealed this is false for 5 years of crypto H1 data:

    SR_min_for_DSR_95 = 2.70 × √52 / √T = 19.5 / √T

| Years | Min Sharpe for DSR 0.95 |
| :--- | :--- |
| 2 | 1.91 |
| 3 | 1.56 |
| 5 | **1.21** |
| 10 | 0.86 |
| 20 | 0.61 |

Even a well-calibrated strategy with Sharpe 0.5 does not pass DSR 0.95
until ~20 years of data. Crypto history is ~13 years for BTC, ~7 for
ETH. The criterion is mathematically infeasible for the available data.

**Therefore, G0_CRITERION_V2** (used for the pivote-A weekly re-run):

```python
G0_CRITERION_V2 = G0Criterion(
    min_sharpe=0.0,
    min_dsr=0.80,  # calibrated to 5y of data
    available_years=5.0,  # 5 years of weekly rebalances
    max_pbo=0.5,
    benchmark_name="buy_and_hold_btc",
    n_trials_for_dsr=4,
    min_sharpe_vs_benchmark=0.10,  # NEW: beat benchmark by 0.10
)
```

The `min_sharpe_vs_benchmark` field is new: it prevents the decision
from becoming "any strategy with Sharpe > 0.34 passes" simply because
Buy&Hold also has Sharpe > 0.34. To claim active edge, an active
strategy must beat the benchmark by at least 0.10 Sharpe.

This is a protocol adjustment after the negative G0-v1. It is
documented here in full, with the mathematical reasoning, so the v2
criterion is auditable and not chosen post-hoc. Any future cycle that
uses v2 must reference this section.

## 7.c. MVP-0 closure — final result

**Status: CLOSED with NEGATIVE gate G0 in three independent tests.**

MVP-0 was evaluated at three gates with progressively stricter setups.
All three point to the same conclusion: no demonstrable alpha in
technical features (price, volume, volatility) at the horizons tested.

### 7.c.1. G0-v1 — daily horizon, 2 years, 3 assets

- Setup: 22 features, daily rebalance, 730 rebalances, 3 assets (BTC,
  ETH, SOL), 2 years of H1 data.
- Result: Sharpe net −0.55 to +0.49 across 4 baselines.
- DSR < 0.36 for all. MinBTL = inf. PBO = 0.003 (stable ranking, but
  no strategy passes DSR or MinBTL).
- **Verdict: NEGATIVE.**

### 7.c.2. G0-v2 — weekly horizon, 5 years, 5 assets

- Setup: 16 features (windowed to weekly scale), weekly rebalance
  (168h), 5 assets (BTC, ETH, SOL, BNB, XRP), 5 years of H1 data,
  5-fold walk-forward.
- Result: Sharpe net −0.01 to +0.49. Random Top2 (0.486) beats
  Momentum (0.444), EqualWeight (0.396), Buy&Hold (0.369).
- DSR max 0.513 (Random). PBO = 0.803 (ranking is unstable).
- **Verdict: NEGATIVE.**

### 7.c.3. Null test — momentum vs 200 random seeds

- Setup: same as G0-v2, but Random Top2 run with 200 different seeds
  to build the null distribution of "pick 2 assets at random".
- Result:
  - Null distribution: mean +0.208, std 0.169, p50 +0.232,
    p95 +0.470.
  - Momentum Sharpe 0.444 at percentile 93.5.
  - **p-value (one-sided) = 0.065** (marginal, not significant at 5%).
  - **Buy&Hold at percentile 84** — 16% of random seeds beat Buy&Hold.
  - **Delta momentum − Buy&Hold = +0.075**, well within the noise band.

### 7.c.4. Honest interpretation

The three tests together say:

1. **The apparent momentum edge over Random (0.236 Sharpe) is mostly
   beta.** Buy&Hold captures 0.161 of that without any signal. The
   delta attributable to the signal (momentum − Buy&Hold) is 0.075.
2. **p = 0.065 is marginal.** With 200 seeds, 13 seeds beat momentum.
   This is "suggestive" at best, not "demonstrated".
3. **PBO = 0.803 means the ranking is unstable.** The winner in-sample
   is not the winner out-of-sample. This is exactly what PBO is
   designed to detect.
4. **The DSR criterion (0.95) is mathematically infeasible** with 5
   years of crypto data even for a well-calibrated strategy (see
   §7.b). Calibrated DSR 0.80 is achieved by zero strategies.

**Conclusion:** With 5 years of crypto H1 data, 5 assets, 16 technical
features, and realistic costs, **no strategy has demonstrable active
alpha over Buy&Hold**. The MVP closes with this result documented.

### 7.c.5. What was NOT established

Important: the negative result is about **this specific setup**, not
about "no edge exists in crypto" or "ML cannot predict crypto
returns". What was NOT established:

- That funding rates, on-chain data, or sentiment carry no signal.
- That multi-horizon ensembles or regime detection add no value.
- That ML models (LightGBM, deep learning) add no value over simple
  momentum.
- That a different universe, timeframe, or cost scenario changes the
  result.

Any of these is a legitimate pivot. The MVP-0 platform supports all of
them without rewriting.

### 7.c.6. Pivots taken

**Pivote A (weekly horizon): executed, negative** (see G0-v2 above).
The 2-year signal (momentum 0.42 vs random 0.035) was a small-sample
artifact. With 5 years, the signal dilutes to p = 0.065 (marginal).

**Pivote B (funding rates): opened after MVP-0 closure.** See
`docs/g0_protocol_v3.md` (created when pivote B is pre-registered).
Funding rates have a plausible economic mechanism (carry / crowding)
and mixed academic evidence, but no guarantee of net edge after
realistic costs (basis, fees, funding flows).

**Decision:** MVP-0 is CLOSED with a negative gate. The platform is
preserved. Pivote B is a new research cycle with its own
pre-registration, not a continuation of MVP-0.

## 8. Cross-references

- Plan §8.15 — original specification of the evaluation protocol.
- Plan §20 — governance documents.
- `docs/significance.md` — the statistical tests.
- `src/locus/evaluation/experiments.py` — the registry.
- `src/locus/evaluation/g0.py` — the criterion and decision logic.
- `docs/g0_protocol.md` — the specific protocol for this cycle
  (created at pre-registration time).
