# Experiment ID conventions

Plan v3.1, §5 (P0 scope) and Anexo B.

## ID format

    <THESIS>-<YYYYMMDD>-<NN>

- `THESIS`: thesis code from the plan (`T1`, `PB`, `A1`, ...)
- `YYYYMMDD`: date the experiment was pre-registered (not the date it ran)
- `NN`: zero-padded sequence (01, 02, ...)

Examples:
- `PB-20261025-01`: first pivote-B pre-registration on 2026-10-25
- `T1-20261101-03`: third T1 pre-registration on 2026-11-01

An ID never changes once assigned. Re-running the same config with new data
or code is a **new** trial with a new ID, not an update.

## Trial registry

File: `trials/registry.jsonl`. JSONL (one object per line), append-only,
versioned in git. Reviewable in PR diffs.

## Schema (plan Anexo B, adapted to JSON)

| Field | Type | Description |
|---|---|---|
| `id` | string | `<THESIS>-<YYYYMMDD>-<NN>` |
| `date` | `YYYY-MM-DD` | Pre-registration date |
| `thesis` | string | `T1`, `PB`, `A1`, ... |
| `hypothesis` | string | One-line config summary |
| `config_hash` | string | `sha256:<hex>` or `pending` |
| `data_hash` | string | `sha256:<hex>` or `pending` |
| `commit` | string | Git SHA where it ran, or `pending` |
| `metrics` | object | Free-form, must include `sharpe` (verdict measurement) |
| `state` | string | `discarded`, `candidate`, `promoted`, `retired` |
| `description` | string | One sentence: what it showed, why that state |

## Hashing

- `config_hash`: SHA-256 of the config dict as canonical JSON (sorted keys,
  no whitespace), prefixed with `sha256:`.
- `data_hash`: SHA-256 of the concatenation of `(path, content)` for each
  input parquet, sorted by path, prefixed with `sha256:`.

Fields marked `pending` must be filled before the trial can be cited as
evidence in a gate (G0-G4).

## Append-only rule

Append new entries; never edit existing lines. If a row is wrong, append a
new row with a new ID and mark the old one `state: "retired"`. The history
is the evidence.
