# This file was generated with the assistance of an AI coding tool.
"""OCCT oracle harness (plan §6).

Golden comparisons against OCCT run **only** where pythonocc-core exists
(ifckit's dev conda env). Skipped decks everywhere else; expected numbers
are carried by the committed corpus fixtures so plain CI needs no OCCT.
The oracle relationship: geometrykit kernel results must match OCCT
within each fixture's tolerance budget — OCCT is the incumbent authority,
never a dependency.
"""

from __future__ import annotations

import pytest

try:
    import OCC.Core.BRep  # noqa: F401

    HAS_OCC = True
except ImportError:
    HAS_OCC = False

requires_occ = pytest.mark.oracle
requires_occ = pytest.mark.skipif(not HAS_OCC, reason="requires pythonocc-core (OCCT oracle)")


def oracle_tolerance(record: dict, default: float = 1e-9) -> float:
    """Per-record tolerance budget (fixture authors may override per corpus entry)."""
    body = record.get("body", {})
    meta = record.get("meta", {}) if isinstance(record.get("meta"), dict) else {}
    return float(meta.get("tol", body.get("tol", default)))
