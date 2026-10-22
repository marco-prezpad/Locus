# Statistical Significance

Tests for whether an observed Sharpe is real or consistent with noise,
given that multiple strategies were tried. Applied at gate G0 to
decide whether the project has demonstrable edge.

## 1. The problem: multiple comparisons

You test 4 strategies. The best one has Sharpe 0.47. Is it real? With
4 independent trials, the expected maximum Sharpe under the null
(everyone is noise) is roughly `1.03 / sqrt(T_years)`. With 2 years of
data, that's ~0.73. Your 0.47 is **below** the null expectation. It's
not significant.

The tools below formalize this:

| Test | Question |
| :--- | :--- |
| **DSR** | P[true Sharpe > 0] after adjusting for N trials |
| **PBO** | P[best in-sample = worst out-of-sample] |
| **MinBTL** | How many years needed for this Sharpe to be significant |
| **Block bootstrap** | 95% CI for Sharpe, preserving autocorrelation |

## 2. Deflated Sharpe Ratio (DSR)

Bailey & López de Prado (2014).

Adjusts the observed Sharpe `SR_hat` for:
- Number of trials `N` (via expected max Sharpe under the null).
- Skewness of returns (fat left tail → lower DSR).
- Kurtosis (fat tails → lower DSR).
- Sample length `T`.

Formula:
```text
SR_0 = sqrt(Var(SR_hat)) * ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(Ne)))
DSR = Φ((SR_hat - SR_0) * sqrt(T-1) / sqrt(1 - γ3·SR_hat + ((γ4-1)/4)·SR_hat²))
```

Where `γ ≈ 0.5772` (Euler-Mascheroni), `γ3` is skewness, `γ4` is
kurtosis.

**Interpretation:**
- DSR > 0.95: strong evidence. Rare when N is large.
- DSR > 0.50: weak evidence.
- DSR ≈ 0.50: indistinguishable from noise.
- DSR < 0.50: worse than the null expectation.

**Typical pattern:** DSR drops dramatically as `N` grows. With N=1 and
Sharpe 0.5 on 2000 obs, DSR ≈ 0.99. With N=100, DSR ≈ 0.30.

## 3. Probability of Backtest Overfitting (PBO)

Bailey, Borwein, López de Prado & Zhu (2017), via CSCV.

Method:
1. Split the returns into S sub-samples (S=16 default).
2. For each of the `C(S, S/2)` combinations of S/2 as in-sample and
   S/2 as out-of-sample:
   - Find the best strategy in-sample.
   - Compute its out-of-sample rank among all strategies.
   - Record it as overfit if the OOS rank is below the median.
3. PBO = fraction of combinations that are overfit.

**Interpretation:**
- PBO = 0.0: no overfitting detected (dominant strategy wins both).
- PBO = 0.5: indistinguishable from random.
- PBO > 0.5: overfitting likely.
- PBO = 1.0: pure overfitting.

**Use:** if the best in-sample strategy has PBO > 0.5, it is probably
overfit. The strategy's expected out-of-sample performance is worse
than median.

## 4. Minimum Backtest Length (MinBTL)

Bailey, Borwein, López de Prado & Zhu (2014).

The minimum number of years of data needed for a given Sharpe to be
distinguishable from the expected max under N trials.

Formula:
```text
MinBTL_years = (E[max SR_N] / SR_target)²
```

Where `E[max SR_N] ≈ Z_N` is the expected maximum of N standard normals
(per year).

**Interpretation:**
- MinBTL = 2.0 years, you have 2.5: the Sharpe is significant.
- MinBTL = 5.0 years, you have 2: not significant yet.
- MinBTL = inf: `SR_target` is below `E[max SR_N]`; no length suffices.

**Rule of thumb:** with N=4 trials and Sharpe 0.5, MinBTL is ~5 years.
With N=100 trials and Sharpe 0.5, MinBTL is ~20 years. Most published
strategies have Sharpe 0.5-1.0 on 10+ years; that's why they survive
the multiple comparison adjustment.

## 5. Block bootstrap Sharpe CI

Standard Sharpe CI assumes iid returns. Real returns are
autocorrelated. Block bootstrap:
1. Divide returns into blocks of `sqrt(T)` (default).
2. Sample blocks with replacement.
3. Compute Sharpe on each resampled series.
4. Take 2.5% and 97.5% quantiles.

The CI is wider than the naive one, often 1.5-2× wider with strong
autocorrelation.

**Interpretation:** if the 95% CI includes 0, the Sharpe is not
significantly different from 0 even without multiple comparison
adjustment.

## 6. Experiment counter

`ExperimentRegistry` tracks every experiment run. Each record has a
`purpose`:
- `exploration`: notebooks, quick tests, no influence on selection.
- `selection`: experiments that informed model/feature/hyperparameter
  choices.
- `final`: the one-shot final evaluation (only one allowed).

`n_trials` (input to DSR) = selection + final. Exploration is excluded
because it did not influence a decision.

**Rule (plan §8.15):** once the final experiment runs, no more
selection experiments are allowed for that phase. Any further change
is a new exploration cycle with its own caveats.

## 7. Application at G0

At gate G0, we run all 4 baselines through F6 and apply the 4 tests
to each. The question: does any baseline have DSR > 0.95 **and**
MinBTL <= the data we have? If yes, there is demonstrable edge. If
no, the result is negative and documented.

**Expected outcome:** with 2 years of data and 4 baselines, no
strategy will pass all four tests. That's the honest answer to "is
there edge?", and it's the input to the pivot decision (§8.15).

## 8. Cross-references

- `docs/backtest.md` — the F6 engine that produces the returns.
- `docs/baselines.md` — the strategies tested.
- Plan §8.15 — final evaluation protocol.
- Plan §21 — documented risks.
- `src/locus/evaluation/significance.py` — implementation.
- `scripts/smoke_f7a1.py` — smoke on real data.
