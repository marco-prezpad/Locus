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

## Conditions for opening G1 (T1 pre-registration)

- Trial registry count reconciles with the actual number of configurations tested, or the discrepancy is documented and accepted.
- P1.c capture running continuously for ≥4 weeks with documented gaps < 1%.
- T1 pre-registration file committed **before** running T1.

**Signed by:** Marco Pérez Padilla on 2026-10-27
