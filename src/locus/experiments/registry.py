"""
registry.py

Trial registry for experiment tracking (plan v3.1, P0 deliverable 2).

Every configuration evaluated is a trial. The registry is the source of
truth for n_trials in DSR and PBO. Without it, any p-value is
overoptimistic (plan §6.3).

Format: JSONL, append-only, versioned in git. One Trial per line.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date as date_type
from pathlib import Path
from typing import Any, Literal

State = Literal["discarded", "candidate", "promoted", "retired"]

_VALID_STATES: frozenset[str] = frozenset({"discarded", "candidate", "promoted", "retired"})

DEFAULT_PATH = Path("trials/registry.jsonl")


@dataclass(frozen=True, slots=True)
class Trial:
    """One configuration evaluated. See docs/conventions.md for the schema."""

    id: str
    date: str
    thesis: str
    hypothesis: str
    config_hash: str
    data_hash: str
    commit: str
    metrics: dict[str, Any] = field(default_factory=dict)
    state: str = "discarded"
    description: str = ""

    def __post_init__(self) -> None:
        try:
            date_type.fromisoformat(self.date)
        except ValueError as exc:
            raise ValueError(f"date must be YYYY-MM-DD, got {self.date!r}") from exc

        parts = self.id.split("-")
        if len(parts) != 3:
            raise ValueError(f"id must be <THESIS>-<YYYYMMDD>-<NN>, got {self.id!r}")
        thesis, datepart, seq = parts
        if not thesis or len(datepart) != 8 or not datepart.isdigit():
            raise ValueError(f"bad id format: {self.id!r}")
        if len(seq) != 2 or not seq.isdigit():
            raise ValueError(f"bad id format: {self.id!r}")

        if self.state not in _VALID_STATES:
            raise ValueError(f"bad state: {self.state!r}")


def hash_config(config: dict[str, Any]) -> str:
    """SHA-256 of canonical JSON (sorted keys, no whitespace)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def hash_data(files: list[Path]) -> str:
    """SHA-256 of (path, content) for each file or directory, sorted by path.

    For directories, all `.parquet` files are collected recursively and
    hashed in sorted order.
    """
    expanded: list[Path] = []
    for f in files:
        if f.is_dir():
            expanded.extend(sorted(f.rglob("*.parquet")))
        else:
            expanded.append(f)
    h = hashlib.sha256()
    for f in sorted(expanded, key=str):
        h.update(str(f).encode())
        h.update(f.read_bytes())
    return "sha256:" + h.hexdigest()


def load(path: Path = DEFAULT_PATH) -> list[Trial]:
    """Load all trials. Returns empty list if file does not exist."""
    if not path.exists():
        return []
    trials: list[Trial] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                trials.append(Trial(**json.loads(stripped)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: invalid trial: {exc}") from exc
    return trials


def append(trial: Trial, path: Path = DEFAULT_PATH) -> None:
    """Append a trial. Fails if the ID already exists."""
    existing = {t.id for t in load(path)}
    if trial.id in existing:
        raise ValueError(f"trial {trial.id!r} already exists in {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(trial), ensure_ascii=False) + "\n")


def count(path: Path = DEFAULT_PATH) -> int:
    """Number of trials in the registry."""
    return len(load(path))
