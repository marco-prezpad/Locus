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

## 8. Cross-references

- `docs/data.md` — the layer F4 consumes from.
- `docs/features.md` — the layer F4 consumes from (F3).
- Plan §8.4 — uniqueness and `n_eff` specification.
- Plan §8.15 — final evaluation protocol and DSR/PBO/MinBTL (F15).
- `src/locus/datasets/` — implementation.
- `tests/unit/datasets/` — tests.
- `scripts/smoke_f4.py` — end-to-end smoke.
