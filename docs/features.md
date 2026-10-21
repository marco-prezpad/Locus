# Feature Layer

Documentation for the feature engineering layer (phase F3). Covers the
`Feature` Protocol, the config-driven `FeatureRegistry`, the
`FeaturePipeline`, all feature families, and the leakage guarantees
that make the feature dataset safe to use for training and backtesting.

The feature layer consumes OHLCV bars from the data layer (see
`docs/data.md`) and produces a DataFrame with the canonical
`BAR_SCHEMA` columns plus one column per feature.

## 1. Contract

A `Feature` is a stateless transform:

```python
class Feature(Protocol):
    @property
    def name(self) -> str: ...
    def compute(self, df: pl.DataFrame) -> pl.DataFrame: ...
```

Contract enforced by the pipeline and the tests:

Input: DataFrame with BAR_SCHEMA columns, sorted by
(asset_id, timestamp), no duplicates, UTC timestamps.

Output: same DataFrame plus exactly one new column named
self.name. All input columns are preserved.

Backward-looking: at row t, the feature's value depends only
on data at timestamps <= t within the same asset. No
cross-asset contamination, no future data, no look-ahead bias.

Nulls at the start: features may produce nulls for the initial
rows of each asset where insufficient history exists (e.g.
return_20 is null for the first 20 bars of each asset). The
pipeline does not drop these nulls; F4 drops them when building
the training dataset.

Idempotent: applying the feature twice to its own output does
not change existing values.

## 2. Pipeline

```python
pipeline = FeaturePipeline.from_specs(specs)
features_df = pipeline.compute(ohlcv_df)
```

FeaturePipeline:

Validates that feature names are unique.

Sorts input by (asset_id, timestamp) before applying features.

Applies features sequentially (each receives the output of the
previous).

Returns the DataFrame with all feature columns added, sorted.

Empty input: if the input has 0 rows, the pipeline returns the
input unchanged (no features computed, no columns added).

## 3. Config

Features are enabled via configs/base.yaml:

```yaml
data:
  features:
    active:
      - {name: return_1, family: return, params: {n: 1}}
      - {name: return_20, family: return, params: {n: 20}}
      - {name: ma_distance_20, family: ma_distance, params: {window: 20}}
      - {name: rolling_vol_20, family: rolling_volatility, params: {window: 20}}
```

Each entry has three fields:

name: the output column name. Must be unique within the
pipeline. Convention: <family>_<param> (e.g. return_20,
ma_distance_50, rolling_vol_20).

family: the feature family. Must match a registered factory.

params: keyword arguments passed to the family factory.

Adding a new feature is one YAML line, no code change. Adding a new
family is one module + one @register_feature decorator.

## 4. Feature families (F3.1)

### 4.1 Returns

return_n at bar t for asset a:

```text
close[t] / close[t-n] - 1
```

Family: return

Params: n (int > 0)

Nulls: first n bars of each asset.

Range: unbounded, typically [-0.5, 0.5] for 1h crypto bars.

Use: short-horizon momentum, mean reversion signals.

### 4.2 Log returns

log_return_n at bar t:

```text
ln(close[t] / close[t-n])
```

Family: log_return

Params: n (int > 0)

Nulls: first n bars of each asset.

Range: same as simple returns for small values, asymmetric for
large.

Use: preferred over simple returns when the downstream model
assumes additive returns (regression, gradient boosting on
log-returns).

### 4.3 Moving average distance

ma_distance_w at bar t:

```text
(close[t] - MA_w[t]) / MA_w[t]
```

where MA_w[t] = mean(close[t-w+1..t]).

Family: ma_distance

Params: window (int > 0)

Nulls: first window-1 bars of each asset (rolling mean needs
window bars; the first is t=window-1).

Range: unbounded, typically [-0.1, 0.1] for 1h crypto bars.

Sign: positive when close is above the MA (uptrend), negative
when below (downtrend).

Note: MA_w[t] includes the current bar. This is deliberate and
matches the standard rolling-mean convention.

### 4.4 Rolling volatility

rolling_vol_w at bar t:

```text
std(returns_1[t-w+1..t])
```

where returns_1[k] = close[k]/close[k-1] - 1.

Family: rolling_volatility

Params: window (int >= 2; std requires at least 2 points)

Nulls: first window bars of each asset (1 for the return,
window-1 for the std).

Units: raw per-bar units (no annualization). For 1h bars,
typical values are 0.001–0.01 (0.1–1% per hour).

Use: volatility regime features, position sizing signals,
normalization denominators.

Why no annualization: features must be comparable across
timeframes. Annualizing (×√8760 for 1h, ×√365 for 1d) would tie the
feature to the timeframe. Downstream models can annualize if needed.

## 5. Default feature set

The 12 features enabled by default in configs/base.yaml:

| Feature | Family | Params | Nulls (per asset) |
|---|---|---|---|
| return_1 | return | n=1 | 1 |
| return_3 | return | n=3 | 3 |
| return_5 | return | n=5 | 5 |
| return_20 | return | n=20 | 20 |
| return_50 | return | n=50 | 50 |
| log_return_1 | log_return | n=1 | 1 |
| log_return_5 | log_return | n=5 | 5 |
| log_return_20 | log_return | n=20 | 20 |
| ma_distance_20 | ma_distance | window=20 | 19 |
| ma_distance_50 | ma_distance | window=50 | 49 |
| rolling_vol_5 | rolling_volatility | window=5 | 5 |
| rolling_vol_20 | rolling_volatility | window=20 | 20 |

With N assets and T bars each, the total null count per feature is
nulls_per_asset × N. With 1 year of 1h data per asset (~8,760 bars),
the worst feature (return_50) loses 50 / 8,760 ≈ 0.6% of rows.
With 7 days of data, it loses 50 / 168 ≈ 30%. This is why the smoke
test uses at least 90 days when validating the pipeline; the 7-day
data from smoke_f2a3.py is only for end-to-end verification.

## 6. Leakage guarantees

The single most important property of the feature layer: no
look-ahead bias. If a feature at t depends on data after t, every
downstream metric (Sharpe, hit rate, PnL) is fabricated.

### 6.1 How leakage is tested

tests/unit/features/test_leakage.py uses a truncation-invariance
check:

Compute features on the full history.

Truncate the input at cutoff.

Recompute features on the truncated input.

Assert that feature values at timestamps <= cutoff are identical.

If a feature uses any future data (e.g. shift(-n),
rolling_mean(..., center=True), future-looking rolling windows,
or cross-asset future contamination), step 4 fails.

### 6.2 How to add a feature without leaking

Use only shift(n), rolling_*, cum_*, diff, and arithmetic
on past values within the same asset.

Never use shift(-n), center=True, .lead(), or any
function that peeks at rows after t.

Always scope per-asset aggregations with .over("asset_id").
Otherwise features from asset A contaminate asset B's rows at the
same timestamp (not look-ahead per se, but a leak of cross-asset
information that breaks the independence assumptions of the
backtest).

Add the new feature to the pipeline that the leakage test exercises.

### 6.3 Cross-sectional leakage (F3.2)

F3.2 adds cross-sectional features (percentile rank, z-score across
assets at the same timestamp). Those need a different leakage test:
asset A's feature at t must not depend on asset B's data after t.
The truncation-invariance test extends naturally: truncate all assets
at the same cutoff and verify all cross-sectional features.

## 7. Adding a new feature

### 7.1 Adding an instance of an existing family

One YAML line:

```yaml
data:
  features:
    active:
      - {name: return_10, family: return, params: {n: 10}}
```

No code change. The registry picks it up at build time.

### 7.2 Adding a new family

Create src/locus/features/<family>.py with a dataclass:

```python
@dataclass(frozen=True, slots=True)
class MyFeature:
    name: str
    # params...

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.sort(["asset_id", "timestamp"]).with_columns(
            pl.col("close").rolling_mean(self.window).over("asset_id").alias(self.name)
        )


@register_feature("my_family")
def _make(name: str, window: int) -> MyFeature:
    return MyFeature(name=name, window=window)
```

Import the module in src/locus/features/__init__.py:

```python
from locus.features import my_family  # noqa: F401 — register
```

Add unit tests in tests/unit/features/test_<family>.py.

Add the family to the leakage test's pipeline (or add a new
parametrized test).

Update this document (section 4).

## 8. Downstream

F4 (labels) consumes the feature dataset and builds the training set:

Drops rows with null features.

Adds target columns (future_return_1h, future_return_4h,
future_return_24h, probability_positive).

Applies purged walk-forward splits with extended purge
(horizon + lookback) and uniqueness weighting.

F5 (baselines) and F8+ (models) consume the F4 dataset.

## 9. Known limitations

No cross-sectional features yet: F3.2 adds percentile rank,
z-score, relative momentum vs market. They need a cross-sectional
leakage test.

No volume features yet: F3.2 adds volume z-score and relative
volume.

No ATR, EMA distance, volatility ratio yet: F3.2.

Raw per-bar units: volatility is not annualized. Downstream
models can annualize if needed.

Single-asset features only: no feature mixes two assets. If a
feature needs another asset's data (e.g. BTC returns as a market
factor), it must go through the cross-sectional path in F3.2, not
through a direct join that could leak.

## 10. Cross-references

docs/data.md — the layer features are computed from.

src/locus/features/base.py — Protocol + registry.

src/locus/features/pipeline.py — pipeline.

src/locus/features/returns.py, trend.py, volatility.py —
feature families.

tests/unit/features/test_leakage.py — the most important tests.

Plan §8 (F3) — original specification.
