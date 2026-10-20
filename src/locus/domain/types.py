"""
types.py

Primitive type aliases used across the domain layer. Explicit NewTypes
(rather than bare `str` or `datetime`) prevent accidental mixing of
identifiers and timestamps in function signatures.

Author: Marco Pérez Padilla
Date:   21-10-2026
"""

from __future__ import annotations

from datetime import datetime
from typing import NewType

AssetId = NewType("AssetId", str)
Timestamp = NewType("Timestamp", datetime)
Currency = NewType("Currency", str)
