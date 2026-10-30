# G0 — Platform reliability gate

**Plan version:** v3.1 §14
**Commit:** `<SHA del merge del PR de P0>` — fill in after merge
**Date:** 2026-10-27
**Decision:** APPROVED

## Criteria (plan §14)

| Criterion | Required | Actual | Pass |
|---|---|---|---|
| Bit-level reproducibility | diff < 1e-9 | `repro diff = 0.000e+00` on `PB-20261025-13` | ✓ |
| Trial registry documented | all trials recorded | 20 entries in `trials/registry.jsonl`; undercount vs plan §6.3 documented in `docs/b5-close.md` §6.3 | ✓ (with caveat) |
| Tests green | `make fix` + `pytest` pass | all unit tests pass | ✓ |
| One test experiment produces report with verdict | run + report + verdict | `reports/PB-20261025-13.md`, verdict NEGATIVE | ✓ |

## Known limitations accepted

1. **Trial undercount.** Plan §6.3 says the real B5 count exceeds 30; the registry documents 20. Follow-up: reconstruct the remaining ~10-15 entries from git log and script history before T1 pre-registration.
2. **Engine formal validation pending.** P6 requires slippage calibrated on real trades. P1.c (microstructure capture) starts now; the calibration is only meaningful after ≥4 weeks of data.
3. **Universe U10 fixed.** Subject to survivorship bias (plan R6). Any experiment using U10 must declare this limitation.

## Scope of this gate

G0 approves the **platform reliability** for research: backtests, walk-forward, validation, registry. It does **not** gate any specific hypothesis.

The path from G0 to G1 (T1 verdict) is:

1. Pre-register T1 (config, criteria, controls, abandonment plan) in a signed commit.
2. Run T1 through the P0 CLI.
3. G1 verdict per plan §7.4 criteria E1-E10, without reinterpretation.

## Note on A4 (not a G1 condition)

Hypothesis A4 (microstructure) is a **separate alternative**, activated only if T1 fails (plan §8.2, D2). It is not a precondition of G1. A4's own pre-registration requires ≥4 weeks of clean P1.c data (plan §9), but T1 does not depend on P1.c.

**Signed by:** Marco Pérez Padilla on 2026-10-27
