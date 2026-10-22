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

## 8. Cross-references

- Plan §8.15 — original specification of the evaluation protocol.
- Plan §20 — governance documents.
- `docs/significance.md` — the statistical tests.
- `src/locus/evaluation/experiments.py` — the registry.
- `src/locus/evaluation/g0.py` — the criterion and decision logic.
- `docs/g0_protocol.md` — the specific protocol for this cycle
  (created at pre-registration time).
