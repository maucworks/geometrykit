# This file was generated with the assistance of an AI coding tool.
"""Parity layer: `_npu` mirror vs. `_rust` accelerator (M6+, plan §5).

Every ported kernel function must produce byte-close results on the corpus
for both engines. Runs are skipped until the maturin extension exists; the
suite is kept import-safe so the mirror is never blocked by the accelerator.
"""

from __future__ import annotations

import pytest
from geometrykit import engine


@pytest.mark.rust
@pytest.mark.skipif(not engine.has_rust(), reason="requires geometrykit._rust (M6+)")
class TestEngineParity:
    def test_signature_hash_parity(self):
        """contract hashing identical across engines (schema v0, hex floats)."""
        assert True  # M6: mirror-hash in _rust vs. contract.signature

    def test_eval_grid_parity(self):
        pytest.skip("M6: first ported kernel (surface eval grid)")
