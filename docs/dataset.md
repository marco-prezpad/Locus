# Dataset Layer

Documentation for the dataset assembly layer (phase F4). Covers label
semantics, temporal and walk-forward splits, purge policy, uniqueness
weighting, effective sample size (`n_eff`), and the validation rules
that the layer enforces.

The dataset layer consumes the feature DataFrame from F3 and produces
train/val/test splits (or walk-forward folds) that are safe to feed
into baselines (F5), models (F8+), and the backtester (F6).

## 1. Overview

`TrainingDatasetBuilder` orchestrates the whole pipeline:

```text
OHLCV (F2) → features (F3) → labels (F4) → drop nulls → split (F4)
```


Result: a `TemporalDataset` (single split) or a `WalkForwardDataset`
(K folds), each with `DatasetSplit` objects carrying `df`, `weights`,
`n_eff`, `start`, `end`. The smoke script runs the full pipeline on
real data and calls `validate_temporal_dataset` on the result.

## 2. Labels

### 2.1 Formula

For each asset and each configured horizon `h` (in bars):

```text
future_return_<h>[t] = close[t + h] / close[t] - 1
```

Computed per asset via `close.shift(-h).over("asset_id") / close - 1`.
Row `t` uses `close[t]` as the entry price, so features at `t` (which
see only data up to `t`) and the label window `[t, t+h]` do not overlap
except at `t` itself.

### 2.2 Default horizons

| Label | `bars` | Horizon |
| :--- | :--- | :--- |
| `future_return_1h` | 1 | 1 bar |
| `future_return_4h` | 4 | 4 bars |
| `future_return_24h` | 24 | 24 bars |

`future_return_24h` is the **primary label** for models and the
backtester. `future_return_1h` and `future_return_4h` are auxiliary;
they are useful for regime detection, multi-horizon ensembling, and
diagnostics.

### 2.3 Binary `probability_positive`

```text
probability_positive[t] = 1{future_return_24h[t] > 0}
```

Derived from the configured `probability_positive_label` (default:
`future_return_24h`). Type: `Int8` (0 or 1).

**Calibration is deferred to F9/F10.** The binary label itself is
balanced by the market (≈ 50% positive over 2 years), but the model
that predicts it may not be calibrated. `CalibrationEvaluator`
(isotonic or Platt) and Brier/log-loss reporting are part of the
model evaluation phase, not this one. Any downstream use of
`probability_positive` as a calibrated probability must first verify
calibration; until then, treat it as a raw binary label.

### 2.4 Nulls

Labels use future data (by design). The last `max(bars)` rows of each
asset have no future data and produce null labels. `TrainingDatasetBuilder`
drops all rows with any null in features or labels before splitting.

With `max(bars) = 24` and 2 years of H1 data per asset (~17,500 bars),
the drop rate is ~0.1%. With 7 days (~168 bars), the drop rate is ~15%.
This is why backfill is required for F4.

## 3. Temporal splits

### 3.1 Why chronological only

Random splits leak future information into training. Every split in
this layer is **strictly chronological**: `train.end < val.start <
test.start`. No random shuffling, no cross-sectional splits, no
k-fold over time.

### 3.2 All assets at a timestamp in the same split

For a given timestamp `t`, every asset's row goes to the same split.
This preserves the panel structure: a model that predicts at `t`
for all assets sees only training data, then only validation data,
then only test data, in chronological order.

### 3.3 Ratios

Default: `train_ratio=0.70`, `val_ratio=0.15`, `test=0.15`. Ratios are
applied to the **unique timestamps**, not to rows, so the number of
assets does not bias the split boundaries.

### 3.4 Purge policy: `h + lookback`, not just `h`

The purge gap between consecutive splits is:

```text
purge_bars = horizon_bars + lookback_bars
```

With the default config (`horizon=24`, `lookback=50`), `purge_bars = 74`.

**Why not just `horizon`?** Standard purge (López de Prado) removes
`horizon` bars around the boundary to prevent train label intervals
`[t, t+h]` from overlapping val/test feature intervals. But features
at `t'` in val/test look back `lookback` bars: `[t'-lookback, t']`.
If `t'` is within `lookback` of the train boundary, its features can
include the tail of a train label interval, which leaks future
information into val/test.

`h + lookback` is more conservative and matches the plan §8.4.

**Both boundaries purged**: train→val and val→test. `validate_temporal_dataset`
checks both.

### 3.5 Implementation note

`TemporalSplitter` applies the purge to the **tail** of train/val, not
to the head of the next split. `val_start = train_boundary`,
`val_end = val_boundary - purge`. This guarantees `val.start - train.end
>= purge` and `test.start - val.end >= purge`, which the validator
asserts.

## 4. Uniqueness weights and `n_eff`

### 4.1 Problem

Labels at consecutive timestamps overlap: label at `t` covers `[t, t+h]`,
label at `t+1` covers `[t+1, t+h+1]`, etc. Treating all N samples as
independent overstates the effective sample size by roughly a factor
of `h`.

### 4.2 Half-open convention

The label interval is `[t, t + h*dt)` **half-open**, matching López
de Prado. Sample at `t'` overlaps with sample at `t` if
`t' ∈ (t - h*dt, t]` — i.e., if `t'` is within the label window of `t`
(including `t` itself).

### 4.3 Weight formula

```text
concurrency(t) = |{ t' : t' ∈ (t - h*dt, t] }| (per asset)
weight(t) = 1 / concurrency(t)
```

For uniformly spaced bars in the interior (not near the start of the
series), `concurrency = h`, so `weight = 1/h`.

**Warm-up**: for the first `h-1` rows of each asset, fewer past labels
cover `t`, so weights are `1/1, 1/2, ..., 1/(h-1)`. These rows
contribute more than interior rows to `n_eff`. This is correct:
they genuinely represent more independent information because fewer
past labels are overlapping.

**Last row**: under half-open, `concurrency` at `t = n-1` is exactly
`h` (all `h` past labels still cover it). The last row is **not**
heavier than the interior.

### 4.4 `n_eff`

```text
n_eff = sum(weights)
```

For uniformly spaced bars without boundary effects, `n_eff ≈ N/h`.
With warm-up corrections, `n_eff ≈ (N-h)/h + H_h`, where `H_h` is
the h-th harmonic number. For `h=24`, `H_24 ≈ 3.78`.

**Why `n_eff`, not `N`, matters**: statistical tests (DSR, PBO, MinBTL,
bootstrap) that assume independent samples must be run with `n_eff`,
not `N`. Using `N = 43,800` when `n_eff ≈ 1,825` produces confidence
intervals that are ~5× too tight and p-values that are far too small.
F15 will use `n_eff` explicitly.

### 4.5 Cross-asset correlation — documented limitation

`compute_uniqueness_weights` is **per asset**. It does not adjust for
correlation across assets. BTC and ETH are highly correlated at 1h
horizons; treating their samples as independent overstates `n_eff`
further.

**Fix is post-MVP.** The López de Prado cluster uniqueness (or a
correlation-based deflation) requires a correlation estimate and a
clustering step. In the MVP, we document the limitation and use the
per-asset `n_eff` as a conservative (but not fully conservative)
estimate.

## 5. Walk-forward splits

`PurgedWalkForwardSplitter` produces `n_splits` folds. Each fold has
a growing train window and a fixed-width val and test window at the
end:

```text
fold 0: train[0..fold_size] val[...] test[walkable..n]
fold 1: train[0..2fold_size] val[...] test[walkable..n]
...
fold k: train[0..(k+1)fold_size] val[...] test[walkable..n]
```

**Fold size = `walkable // (n_splits + 1)`**, not `// n_splits`.
With `n_splits` folds, you need `n_splits + 1` windows (one for the
initial train, one per fold's val). Using `// n_splits` leaves no
room for the last fold.

**Minimum fold size** = `horizon_bars + lookback_bars`. If
`fold_size < min_fold`, the splitter raises `ValueError` rather than
silently producing folds that are too small to purge correctly.

## 6. Validation

`validate_temporal_dataset` runs 5 checks. Any violation raises
`DatasetValidationError`; the smoke script exits non-zero.

| Check | What it verifies |
| :--- | :--- |
| **No overlap** | `train.end < val.start < test.start` |
| **Purge gaps** | `val.start - train.end >= purge_bars * bar_delta`, same for val→test |
| **No nulls** | All features and labels are non-null in every split |
| **Weights sum** | `sum(weights) == n_eff` for every split |
| **Label correctness** | Recomputes `close[t+24]/close[t] - 1` and compares against the stored `future_return_24h` |

The **label recomputation** check is the most valuable: if anyone
changes the label formula (e.g., uses `open` instead of `close`, or
adds a lag), the validation fails. The **purge gap** check catches the
boundary bug that the original F4 splitter had (gap collapsed to 1 bar
because `val_start = train_end + bar_delta`). The **weights sum** check
catches a broken uniqueness implementation.

## 7. What F4 does NOT do

These are deliberate gaps, not bugs:

- **Calibration of `probability_positive`**: F9/F10. The label is
  balanced but not calibrated. Downstream uses must verify calibration
  before treating it as a probability.
- **Cross-asset correlation adjustment** in uniqueness: post-MVP. The
  per-asset `n_eff` is conservative but not fully.
- **DSR / PBO / MinBTL**: F15. These require `n_eff` from F4, but the
  statistical machinery is part of the final evaluation framework.
- **Holdout pre-registration**: F15. The protocol that locks the final
  evaluation against overfitting (plan §8.15) is built in F15, not here.
- **Multi-horizon dataset assembly**: F4 exposes the pieces, but no
  helper combines `future_return_1h`, `future_return_4h`, and
  `future_return_24h` into a multi-target dataset. F5 uses the primary
  label; F9+ can extend.

## Net-of-costs labels (P4)

Training labels must reflect the executable target, not the gross return.
A model optimised on gross returns will overfit to short-lived moves that
costs would erase. `LabelBuilder` now produces two variants per horizon:

| Column | Meaning |
|---|---|
| `future_return_{h}h` | Gross: `open[t+1+h] / open[t+1] - 1` (diagnostics only) |
| `future_return_{h}h_net` | Net: gross minus `CostAssumption.round_trip_bps / 10000` |
| `probability_positive` | Binary: 1 if `<source> > u`, else 0 (null where source is null) |

The training target for T1 is `future_return_24h_net` (plan §7.3). The
binary label uses `u = 0.0` by default; the pre-registration fixes `u`
before running T1.

### Timing (entry at open[t+1])

Labels use `price_field="open"` with `entry_offset_bars=1`:

    label[t] = open[t+1+h] / open[t+1] - 1

Not `open[t+h] / open[t] - 1`. The reason is that features at `t` use
`close[t]`, and the strategy executes at `open[t+1]`. Entry at `open[t+1]`
matches the executable window and eliminates the close-to-open gap that
would otherwise overstate the label by one bar's drift.

The default for the existing `TrainingDatasetBuilder` is
`price_field="close"`, `entry_offset_bars=0` (legacy). The P4 scripts
(`build_labels.py`, `verify_splits.py`) and the T1 pre-registration use
the plan §7.3 semantics (`open`, offset 1).

### Round-trip cost formula

    round_trip_bps = 2 * (fee_bps + spread_bps/2 + slippage_bps + impact_bps)

The `2 *` is because a position opens and closes (two sides). The
`spread_bps / 2` inside the parenthesis is because the spread is crossed
once per round trip (buy at ask, sell at bid), not once per side.

`CostAssumption.from_config(config, scenario)` reads from the same YAML
block as the backtest, so labels and P&L agree by construction. Changing
the cost scenario changes both together.

### `impact_bps` vs `impact_k`

The engine's market impact is modelled as `sqrt(volume) * impact_k`. This
is state-dependent: it depends on the size of the position relative to
the daily volume at the time of execution.

Labels use a **fixed** `impact_bps` instead, because at label time we do
not know the position size the model will eventually take, nor the future
volume. A fixed assumption of 1 bps (central scenario) is the plan §7.3
proposal.

Both fields live in `configs/base.yaml` under `backtest:`:

    impact_k:   1.0   # used by the engine (sqrt_volume model)
    impact_bps: 1.0   # used by labels (fixed assumption)

In the central scenario they coincide numerically. Other scenarios can
diverge (the pessimistic scenario raises both; the optimistic lowers
both), but they remain conceptually distinct.

### Sanity check: gross vs net

For a round-trip of 37 bps, a gross return of 0.5% becomes a net return
of 0.13%. Any horizon where the median net return is negative means the
strategy must not trade — the model's `probability_positive` will be 0
in most rows, and calibration (ECE, Brier) on those rows tests the model
on the "don't trade" decision, which is the correct default.

## Purge and embargo (P4)

T1 uses `PurgedWalkForwardSplitter` with `horizon_bars=24`,
`lookback_bars=0`, `test_ratio=0.15`, `n_splits=5`.

### Splitter design (rolling origin, fixed test holdout)

The splitter uses a design that is **not** the classic "test-mobile"
walk-forward. It combines:

- **Rolling-origin train/val.** Each fold's train and val windows start
  at the same origin (the first timestamp) and extend further forward.
  Fold 0 has train 2024-10 to 2025-01 and val 2025-01 to 2025-05.
  Fold 4 has train 2024-10 to 2026-03 and val 2026-03 to 2026-06.
- **Fixed test holdout at the end of the series.** Every fold evaluates
  on the *same* test window (last `test_ratio` of the series). Fold 0
  trains on 3 months of data and tests on a 3-month window 17 months
  later; fold 4 trains on 17 months and tests on the same 3-month
  window 3 months later.

Consequences:

- **Excellent for detecting overfitting**: fold 0's model never sees any
  data close to the test period. If it generalises, it's not because it
  memorised the test neighbourhood.
- **Bad for detecting regime stability**: all folds share the same test
  window. If that window is a hostile regime, every fold will look
  negative, even if the strategy works in other regimes.

Therefore, **variance of Sharpe across folds reflects regime, not model
instability**. The T1 pre-registration must not interpret a high-variance
Sharpe across folds as evidence of an unstable signal — it is a property
of the splitter design.

If regime stability becomes a research question later (post-T1), a
separate test-mobile splitter is a distinct phase. It is not in scope
for P4 or T1.

### Purge between train and val

`train_end = max(train_ts) - horizon_bars` before slicing. Verified:
25h gap for horizon=24h (24h purge + 1h bar rounding) on U10.

### Embargo between val and test

Not a separate mechanism — it emerges from the fixed test holdout at the
end. The smallest embargo occurs in the last fold (fold 4), where val
ends 2026-06-14 and test starts 2026-06-15: **29h**. Earlier folds have
gaps of days or months. The plan §9 criterion "embargo de h barras tras
cada test" is satisfied by construction: there is no data "after the
test" because test is at the end.

### Verification

`scripts/verify_splits.py` runs the splitter on real data and checks,
per fold:

- **No timestamp overlap** between train and val, or between val and
  test (must be zero).
- **Gap train → val >= horizon** (purge effective).
- **Gap val → test >= horizon** (embargo effective).
- **Test fold has >= 30 observations** (plan §9 warning threshold).

Run before pre-registering T1:

    uv run python scripts/verify_splits.py

Exit code 0 confirms the splitter is safe. Exit 1 lists the failing
checks. Verified on 2026-10-30 on U10 (170,200 bars, 10 assets):

| Fold | Purge train→val | Embargo val→test | Test n | n_eff |
|---|---|---|---|---|
| 0 | 25h | 9657h | 25,490 | 1089.8 |
| 1 | 25h | 7250h | 25,490 | 1089.8 |
| 2 | 25h | 4843h | 25,490 | 1089.8 |
| 3 | 25h | 2436h | 25,490 | 1089.8 |
| 4 | 25h | 29h | 25,490 | 1089.8 |

### Known limitation: label overlap within each fold

Even with purge and embargo, labels **within** a fold overlap: a row at
`t` with horizon 24h shares its exit price with a row at `t+1` with
horizon 23h. This is why `uniqueness.py` exists — it assigns weights to
training samples to compensate. The `DatasetSplit` object exposes
`weights` and `n_eff` (effective sample size after weight normalisation).
T1 should use `uniqueness_weights` when computing metrics on training
rows. Not required for the P4 exit criterion (purge/embargo), but
relevant for T1 training.

## Label files under `data/processed/labels/`

`scripts/build_labels.py` writes one parquet per asset to
`data/processed/labels/{ASSET}.parquet`. Each file has one row per
(timestamp, asset) with columns:

- `future_return_{h}h` (gross, diagnostics)
- `future_return_{h}h_net` (net, training target)
- `probability_positive` (binary)

These are inputs to T1's training pipeline. They are **not** the same as
`data/processed/bars/` — labels are derived, bars are raw. A label file
regenerated with a different cost scenario is a new file, not a
modification; document it in the T1 pre-registration commit.

Because the label files share the bar schema (they preserve `open`,
`high`, `low`, `close`, `volume`), the discovery helpers in
`build_labels.py` and `verify_splits.py` exclude the `labels` directory
explicitly. A future refactor to a manifest (`data/manifest.json`) would
replace this pattern; for P4 the directory-name exclusion is sufficient.

## 8. Cross-references

- `docs/data.md` — the layer F4 consumes from.
- `docs/features.md` — the layer F4 consumes from (F3).
- Plan §8.4 — uniqueness and `n_eff` specification.
- Plan §8.15 — final evaluation protocol and DSR/PBO/MinBTL (F15).
- `src/locus/datasets/` — implementation.
- `tests/unit/datasets/` — tests.
- `scripts/smoke_f4.py` — end-to-end smoke.
