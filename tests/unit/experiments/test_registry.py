"""Tests for the trial registry (P0 + coverage additions)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from locus.experiments.registry import (
    Trial,
    append,
    count,
    hash_config,
    hash_data,
    load,
)


def _trial(trial_id: str = "PB-20261025-01", **overrides: object) -> Trial:
    base: dict[str, object] = {
        "id": trial_id,
        "date": "2026-10-25",
        "thesis": "PB",
        "hypothesis": "funding_zscore_30d, k=4, 168h",
        "config_hash": "sha256:abc",
        "data_hash": "sha256:def",
        "commit": "abc1234",
        "metrics": {"sharpe": -0.224},
        "state": "discarded",
        "description": "test",
    }
    base.update(overrides)
    return Trial(**base)  # type: ignore[arg-type]


# --- hashing ---


def test_hash_config_deterministic_regardless_of_key_order() -> None:
    a = hash_config({"k": 4, "feature": "zscore"})
    b = hash_config({"feature": "zscore", "k": 4})
    assert a == b
    assert a.startswith("sha256:")
    assert len(a) == len("sha256:") + 64


def test_hash_config_changes_with_value() -> None:
    assert hash_config({"k": 4}) != hash_config({"k": 5})


def test_hash_data_deterministic_across_order(tmp_path: Path) -> None:
    f1 = tmp_path / "a.parquet"
    f2 = tmp_path / "b.parquet"
    f1.write_bytes(b"content-a")
    f2.write_bytes(b"content-b")
    assert hash_data([f1, f2]) == hash_data([f2, f1])


def test_hash_data_expands_directory(tmp_path: Path) -> None:
    d = tmp_path / "sub"
    d.mkdir()
    (d / "x.parquet").write_bytes(b"x")
    (d / "y.parquet").write_bytes(b"y")
    (d / "ignore.txt").write_bytes(b"not-hashed")
    h = hash_data([d])
    assert h.startswith("sha256:")
    (d / "x.parquet").unlink()
    assert hash_data([d]) != h


def test_hash_data_single_file(tmp_path: Path) -> None:
    f = tmp_path / "a.parquet"
    f.write_bytes(b"content-a")
    h = hash_data([f])
    assert h.startswith("sha256:")
    f.write_bytes(b"different")
    assert hash_data([f]) != h


def test_hash_data_mixed_file_and_dir(tmp_path: Path) -> None:
    f = tmp_path / "single.parquet"
    f.write_bytes(b"single")
    d = tmp_path / "sub"
    d.mkdir()
    (d / "x.parquet").write_bytes(b"x")
    h = hash_data([f, d])
    assert h.startswith("sha256:")


def test_hash_data_empty_directory(tmp_path: Path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    h = hash_data([d])
    assert h == "sha256:" + hashlib.sha256(b"").hexdigest()


# --- append / load / count ---


def test_append_and_load(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    append(_trial("PB-20261025-01"), path)
    append(_trial("PB-20261025-02", hypothesis="other"), path)
    loaded = load(path)
    assert [t.id for t in loaded] == ["PB-20261025-01", "PB-20261025-02"]


def test_append_duplicate_id_fails(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    append(_trial("PB-20261025-01"), path)
    with pytest.raises(ValueError, match="already exists"):
        append(_trial("PB-20261025-01"), path)


def test_append_creates_missing_parents(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "c" / "registry.jsonl"
    assert not path.parent.exists()
    append(_trial("PB-20261025-01"), path)
    assert path.exists()


def test_count(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    assert count(path) == 0
    append(_trial("PB-20261025-01"), path)
    assert count(path) == 1


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load(tmp_path / "missing.jsonl") == []


def test_round_trip_preserves_data(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    t = _trial(
        "PB-20261025-01",
        metrics={"sharpe": 0.5, "n": 10, "nested": {"x": 1}},
    )
    append(t, path)
    assert load(path)[0] == t


def test_load_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    t = _trial("PB-20261025-01")
    line = json.dumps(asdict(t))
    line2 = line.replace("PB-20261025-01", "PB-20261025-02")
    path.write_text(f"{line}\n\n\n{line2}\n", encoding="utf-8")
    loaded = load(path)
    assert [x.id for x in loaded] == ["PB-20261025-01", "PB-20261025-02"]


# --- validation ---


def test_bad_date() -> None:
    with pytest.raises(ValueError, match="date"):
        _trial(date="2026/10/25")


def test_id_two_parts() -> None:
    with pytest.raises(ValueError, match="id"):
        _trial("PB-20261025")


def test_id_four_parts() -> None:
    with pytest.raises(ValueError, match="id"):
        _trial("PB-20261025-01-extra")


def test_id_empty_thesis() -> None:
    with pytest.raises(ValueError, match="id"):
        _trial("-20261025-01")


def test_id_short_datepart() -> None:
    with pytest.raises(ValueError, match="id"):
        _trial("PB-2026102-01")


def test_id_non_numeric_datepart() -> None:
    with pytest.raises(ValueError, match="id"):
        _trial("PB-2026102A-01")


def test_id_long_seq() -> None:
    with pytest.raises(ValueError, match="id"):
        _trial("PB-20261025-001")


def test_id_non_numeric_seq() -> None:
    with pytest.raises(ValueError, match="id"):
        _trial("PB-20261025-AA")


def test_bad_state() -> None:
    with pytest.raises(ValueError, match="state"):
        _trial(state="not-a-state")


def test_invalid_jsonl_raises(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    path.write_text("{not valid json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid trial"):
        load(path)


def test_load_invalid_trial_fields(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    bad = {
        "id": "not-a-valid-id",
        "date": "2026-10-25",
        "thesis": "PB",
        "hypothesis": "h",
        "config_hash": "sha256:x",
        "data_hash": "sha256:y",
        "commit": "abc",
        "metrics": {},
        "state": "discarded",
        "description": "",
    }
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid trial"):
        load(path)
