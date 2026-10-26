# Trial registry

Every configuration evaluated in this project goes here, even if discarded
(plan v3.1 §3 and §6.3). This file is the source of truth for `n_trials` in
DSR and PBO. Without it, every p-value is overoptimistic.

## Adding a trial

```python
from pathlib import Path
from locus.experiments.registry import Trial, append, hash_config, hash_data

trial = Trial(
    id="PB-20261025-04",
    date="2026-10-25",
    thesis="PB",
    hypothesis="funding_zscore_30d, k=4, 168h/168h, central costs",
    config_hash=hash_config({"feature": "funding_zscore_30d", "k": 4, "h": 168}),
    data_hash=hash_data(
        [
            Path("data/processed/bars/"),
            Path("data/processed/funding/"),
        ]
    ),
    commit="abc1234",
    metrics={"sharpe": -0.224, "return": -0.1056, "folds_pos": "1/3"},
    state="discarded",
    description="Fold 0 +0.55, fold 1 -0.49, fold 2 -1.76. Regime shift.",
)
append(trial)
```

The `sharpe` field in `metrics` must be the **walk-forward, event-driven engine** value, not the vectorized one. The vectorized can be included as
`sharpe_vectorized`, but the verdict measurement is the engine.

## Reconstructing pivote B
The seed entries below come from session logs. Hashes and commit SHAs are
marked `pending` because the B5 runs predate the registry. They are filled
when B5 is re-executed from config (P0 deliverable 7).

## Do not edit existing rows
Append-only. If a row is wrong, append a corrected row with a new ID and
mark the old one `state: "retired"`.
