"""
experiments.py

ExperimentRegistry: persistent counter of experiments for the DSR
`n_trials` input and for the governance protocol (plan §8.15).

Each record carries `purpose`:
- `exploration`: notebooks, quick tests, no influence on selection.
- `selection`: experiments that informed the choice of model, features,
  hyperparameters, or thresholds.
- `final`: the one-shot final evaluation (only one allowed per phase).

The registry is JSON-backed and appends atomically. It is read by F7's
report to compute `n_trials = selection + final` (exploration is
excluded; if it did not influence a decision, it is not an alternative
for DSR's purpose).

Author: Marco Pérez Padilla
Date:   22-10-2026
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

Purpose = Literal["exploration", "selection", "final"]


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    run_id: str
    purpose: Purpose
    strategy: str
    metrics: dict[str, float] = field(default_factory=dict)
    config_hash: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.purpose not in ("exploration", "selection", "final"):
            raise ValueError(f"invalid purpose: {self.purpose}")


class ExperimentRegistry:
    """Persistent list of experiment records."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._records: list[ExperimentRecord] = []
        if self._path.exists():
            self._load()

    def __len__(self) -> int:
        return len(self._records)

    def all(self) -> list[ExperimentRecord]:
        return list(self._records)

    def add(
        self,
        purpose: Purpose,
        strategy: str,
        metrics: dict[str, float] | None = None,
        config_hash: str = "",
    ) -> ExperimentRecord:
        if purpose == "final":
            existing = [r for r in self._records if r.purpose == "final"]
            if existing:
                raise ValueError(
                    f"only one 'final' experiment is allowed; found {len(existing)} already"
                )
        record = ExperimentRecord(
            run_id=str(uuid.uuid4()),
            purpose=purpose,
            strategy=strategy,
            metrics=dict(metrics or {}),
            config_hash=config_hash,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._records.append(record)
        self._save()
        return record

    def count(self, purpose: Purpose | None = None) -> int:
        if purpose is None:
            return len(self._records)
        return sum(1 for r in self._records if r.purpose == purpose)

    def n_trials(self) -> int:
        """Number of trials for DSR: selection + final (exploration
        excluded). Minimum 1 to keep DSR defined."""
        return max(self.count("selection") + self.count("final"), 1)

    def _load(self) -> None:
        with open(self._path, encoding="utf-8") as f:
            data = json.load(f)
        self._records = [ExperimentRecord(**rec) for rec in data.get("records", [])]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"records": [asdict(r) for r in self._records]},
                f,
                indent=2,
            )
        tmp.replace(self._path)
