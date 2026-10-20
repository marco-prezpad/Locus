"""
test_isolation.py

Ensures the domain layer is pure Python: it must not import third-party
libraries (pandas, numpy, sklearn, binance, sqlalchemy, ...).

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from pathlib import Path

import pytest

FORBIDDEN_MODULES = (
    "pandas",
    "polars",
    "numpy",
    "pyarrow",
    "scipy",
    "sklearn",
    "lightgbm",
    "xgboost",
    "torch",
    "transformers",
    "mlflow",
    "sqlalchemy",
    "binance",
    "ccxt",
    "requests",
    "httpx",
    "streamlit",
    "plotly",
)

DOMAIN_DIR = Path(__file__).resolve().parents[3] / "src" / "locus" / "domain"
DOMAIN_FILES = sorted(DOMAIN_DIR.glob("*.py"))


def test_domain_directory_exists() -> None:
    assert DOMAIN_DIR.is_dir(), f"missing domain dir: {DOMAIN_DIR}"
    assert DOMAIN_FILES, "no files found in domain layer"


@pytest.mark.parametrize("path", DOMAIN_FILES, ids=lambda p: p.name)
def test_no_forbidden_imports(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    for module in FORBIDDEN_MODULES:
        assert f"import {module}" not in source, f"{path.name} imports forbidden module {module!r}"
        assert f"from {module}" not in source, (
            f"{path.name} imports from forbidden module {module!r}"
        )
