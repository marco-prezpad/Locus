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

## 4. Feature families

### 4.1 Returns (per-asset)

`return_n` at bar `t` for asset `a`:

```text
close[t] / close[t-n] - 1
```


- **Family**: `return`. **Params**: `n` (int > 0).
- **Nulls**: first `n` bars of each asset.
- **Use**: short-horizon momentum, mean reversion signals.

### 4.2 Log returns (per-asset)

`log_return_n` at bar `t`: `ln(close[t] / close[t-n])`.

- **Family**: `log_return`. **Params**: `n` (int > 0).
- **Nulls**: first `n` bars of each asset.
- **Use**: additive returns for regression / GBM targets.

### 4.3 Moving average distance (per-asset)

`ma_distance_w[t] = (close[t] - MA_w[t]) / MA_w[t]`,
`MA_w[t] = mean(close[t-w+1..t])`.

- **Family**: `ma_distance`. **Params**: `window` (int > 0).
- **Nulls**: first `window - 1` bars (rolling mean needs `window`).
- **Note**: `MA_w[t]` includes the current bar. Deliberate; matches
  standard rolling-mean convention.

### 4.4 Rolling volatility (per-asset)

`rolling_vol_w[t] = std(returns_1[t-w+1..t])`,
`returns_1[k] = close[k]/close[k-1] - 1`.

- **Family**: `rolling_volatility`. **Params**: `window` (int >= 2).
- **Nulls**: first `window` bars.
- **Units**: raw per-bar units (no annualization), comparable across
  timeframes.

### 4.5 EMA distance (per-asset)

`ema_distance_w[t] = (close[t] - EMA_w[t]) / EMA_w[t]`.

- **Family**: `ema_distance`. **Params**: `window` (int > 0).
- **Nulls**: warm-up, roughly `window - 1` bars.
- **Note**: EMA weights recent bars more heavily than the SMA. Useful
  when recent trend matters more than average trend. Uses
  `adjust=False` (recursive EMA, standard for features).

### 4.6 ATR as percentage of close (per-asset)

`atr_pct_w[t] = ATR_w[t] / close[t]`,
`TR[t] = max(high[t]-low[t], |high[t]-prev_close[t]|, |low[t]-prev_close[t]|)`,
`ATR_w[t] = mean(TR[t-w+1..t])`.

- **Family**: `atr_percent`. **Params**: `window` (int >= 2).
- **Nulls**: first `window - 1` bars.
- **Range**: typically `0.001–0.05` for 1h crypto bars.
- **Note**: raw ATR (in price units) is useless across assets because
  prices range from `0.00001` (SHIB) to `100000` (BTC). Percentage of
  close makes it comparable.

### 4.7 Volatility ratio (per-asset)

`vol_ratio_s_l[t] = rolling_vol_s[t] / rolling_vol_l[t]`.

- **Family**: `volatility_ratio`. **Params**: `short_window` (int >= 2),
  `long_window` (int > short_window).
- **Nulls**: first `long_window` bars.
- **Interpretation**: >1 means recent vol is above longer-term vol
  (regime shift), <1 below.
- **Null when `long_vol = 0`**: constant price produces 0/0; we
  return null.

### 4.8 Volume z-score (per-asset)

`volume_zscore_w[t] = (volume[t] - rolling_mean_w[t]) / rolling_std_w[t]`.

- **Family**: `volume_zscore`. **Params**: `window` (int >= 2).
- **Nulls**: first `window - 1` bars, plus wherever std is 0
  (constant volume in the window).
- **Use**: detect volume spikes and dry-ups. More sensitive to outliers
  than relative volume.

### 4.9 Relative volume (per-asset)

`relative_volume_w[t] = volume[t] / rolling_mean_w[t]`.

- **Family**: `relative_volume`. **Params**: `window` (int > 0).
- **Nulls**: first `window - 1` bars, plus wherever mean is 0.
- **Use**: similar to z-score but robust to heavy-tailed volume
  distributions. Values >1 mean above-average, <1 below.

### 4.10 Cross-sectional features

Cross-sectional features rank or normalize an existing per-asset
column **across all assets at the same timestamp**. They require their
`source` column to be present in the DataFrame before they run; the
pipeline enforces ordering.

**Sign convention**: higher source value → higher rank / positive
z-score. For "lower is better" sources (volatility), negate the
source or invert the interpretation downstream.

#### 4.10.1 Percentile rank

`xs_percentile_<source>[t] = (rank(source[t]) - 0.5) / n_non_null`,
where `rank` uses method="average" (ties get the same rank) and
`n_non_null` is the number of assets with non-null source at `t`.

- **Family**: `xs_percentile_rank`. **Params**: `source` (column name).
- **Range**: `(0, 1)`, centered at `0.5`.
- **Null**: where source is null.
- **Use**: compare an asset's momentum/volatility to its peers. A
  value close to 1 means "best in class at this timestamp".

#### 4.10.2 Cross-sectional z-score

`xs_zscore_<source>[t] = (source[t] - mean(source[t])) / std(source[t])`,
computed across assets at the same timestamp.

- **Family**: `xs_zscore`. **Params**: `source` (column name).
- **Null**: where source is null, where fewer than 2 non-null assets
  exist at `t`, or where std is 0 (all tied).
- **Use**: similar to percentile rank but preserves magnitude
  information.

#### 4.10.3 Relative momentum vs market

`rel_momentum_<source>[t] = source[t] - mean(source[t])`,
computed across assets at the same timestamp.

- **Family**: `relative_momentum`. **Params**: `source` (column name).
- **For `source=return_20`**: asset's 20-bar momentum minus the
  equal-weighted market momentum. Positive = outperforming.
- **Null**: where source is null.
- **Use**: simplest long-short signal; the market component cancels out.

**Why equal-weighted market**: no market-cap data available in the
MVP. Equal-weight is the simplest defensible choice; market-cap
weighting lands when indices are supported (B9).

### 4.11 Funding features (B.2)

Computed on the funding DataFrame (`asset_id`, `venue`, `symbol`,
`timestamp`, `funding_rate`), not on the price panel. Per-asset
features group by `asset_id`; cross-sectional features group by
`timestamp` after `normalize_funding_timestamps` (floors each
timestamp to the nearest minute to absorb 2-3ms drift observed in
production).

| Feature | Family | Formula | Nulls |
| :--- | :--- | :--- | :--- |
| `funding_zscore_30d` | funding_zscore | rolling z-score, window 90 | 89 warm-up |
| `funding_zscore_7d` | funding_zscore | rolling z-score, window 21 | 20 warm-up |
| `funding_extreme_30d` | funding_extreme | `1{|zscore_30d| > 2}` | inherits source |
| `funding_trend_7d` | funding_trend | `rate[t] − rate[t−21]` | 21 warm-up |
| `funding_xs_rank` | funding_xs_rank | percentile rank across assets, (0,1) | 0 (5 assets) |
| `funding_xs_zscore` | funding_xs_zscore | cross-sectional z-score | 0 (5 assets) |

Leakage test: `tests/unit/features/test_funding_leakage.py` (truncation
invariance). Cross-sectional alignment test:
`test_xs_rank_aligned_with_drift` (2ms and 4ms drift on two assets
must produce aligned ranks after normalization).

**Null contract for z-score features (`funding_zscore_*`, `funding_xs_zscore`):**

- `null` during warm-up (incomplete rolling window) or where the input is null.
- `0.0` when the rolling / cross-sectional standard deviation is exactly zero
  (constant funding rate within the window or across assets). A constant rate
  carries no dispersion information; returning `0.0` (neutral) instead of `null`
  avoids a selection bias: calm markets (funding pinned at the venue default,
  e.g. Binance `0.0001`) would otherwise be systematically excluded when nulls
  are dropped before training.

## 5. Default feature set

The 22 features enabled by default in `configs/base.yaml`:

| Feature | Family | Params | Nulls (per asset) |
| :--- | :--- | :--- | :--- |
| `return_1` | return | n=1 | 1 |
| `return_3` | return | n=3 | 3 |
| `return_5` | return | n=5 | 5 |
| `return_20` | return | n=20 | 20 |
| `return_50` | return | n=50 | 50 |
| `log_return_1` | log_return | n=1 | 1 |
| `log_return_5` | log_return | n=5 | 5 |
| `log_return_20` | log_return | n=20 | 20 |
| `ma_distance_20` | ma_distance | window=20 | 19 |
| `ma_distance_50` | ma_distance | window=50 | 49 |
| `ema_distance_20` | ema_distance | window=20 | ~19 |
| `ema_distance_50` | ema_distance | window=50 | ~49 |
| `rolling_vol_5` | rolling_volatility | window=5 | 5 |
| `rolling_vol_20` | rolling_volatility | window=20 | 20 |
| `atr_pct_14` | atr_percent | window=14 | 13 |
| `vol_ratio_5_20` | volatility_ratio | short=5, long=20 | 20 |
| `volume_zscore_20` | volume_zscore | window=20 | 19 |
| `relative_volume_20` | relative_volume | window=20 | 19 |
| `xs_percentile_return_20` | xs_percentile_rank | source=return_20 | inherits 20 |
| `xs_zscore_return_20` | xs_zscore | source=return_20 | inherits 20 |
| `xs_percentile_ma_distance_20` | xs_percentile_rank | source=ma_distance_20 | inherits 19 |
| `rel_momentum_vs_market_20` | relative_momentum | source=return_20 | inherits 20 |

Cross-sectional features inherit their source's nulls (they are null
wherever the source is null). They may also be null at timestamps where
fewer than 2 non-null assets exist, or where the cross-section is all
tied (z-score only).

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

### 6.3 Cross-sectional leakage

Cross-sectional features (`xs_percentile_rank`, `xs_zscore`,
`relative_momentum`) see all assets at timestamp `t`. The concern is
that asset A's feature at `t` might depend on asset B's data after `t`.

`tests/unit/features/test_leakage_cross_sectional.py` covers two
scenarios:

1. **Truncation invariance across assets**: compute features on the
   full panel, truncate all assets at the same `cutoff`, recompute on
   the truncated panel, and verify values at `t <= cutoff` are
   identical.
2. **Anomaly isolation**: inject a massive price spike in asset A at
   the last bar. Asset B's cross-sectional features at earlier
   timestamps must be unchanged. This catches implementations that
   accidentally aggregate over the whole panel (e.g. `pl.col(...).mean()`
   without `.over("timestamp")`), which would leak the anomaly backward.

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

- **Equal-weighted cross-section**: no market-cap data; market proxy
  is the equal-weighted mean of the asset universe at each timestamp.
  Market-cap-weighted features will be available once indices are
  supported (B9).
- **No multi-asset joins by hand**: features that need another asset's
  data must go through the cross-sectional path. A raw join across
  assets risks leaking future data from one asset into another's row.
- **Cross-sectional nulls at the start**: percentile rank / z-score
  are undefined for the first rows where too few assets have enough
  history. Downstream (F4) drops null rows.
- **Raw per-bar volatility units**: not annualized; models can
  annualize downstream.
- **Ties in percentile rank**: method="average" gives all tied assets
  the same rank, so a flat cross-section yields 0.5 for all.
- **Z-score nulls on flat cross-sections**: if all assets have the
  same source value at a timestamp, std=0, z-score is null.

## 10. Cross-references

docs/data.md — the layer features are computed from.

src/locus/features/base.py — Protocol + registry.

src/locus/features/pipeline.py — pipeline.

src/locus/features/returns.py, trend.py, volatility.py —
feature families.

tests/unit/features/test_leakage.py — the most important tests.

Plan §8 (F3) — original specification.
