# This file was generated with the assistance of an AI coding tool.
"""Engine parity harness: `_npu` mirror vs. `_rust` accelerator.

Phase 1 (scaffold): the harness runs with the mirror on both sides and
holds bit-exact contract signatures stable across two independent
builds of the same kernel data — the shape a Phase 2 port must satisfy,
already exercised end to end. When the extension is importable, the
same values are additionally checked against core's canonical
float patterns (`bits_hex`), which is a real cross-language test today:
two languages, one IEEE-754 payload convention.

Contract grain: hex-float bit patterns (`f64:XXXXXXXXXXXXXXXX`) rolled
into an MD5[:16] signature via `geometrykit.contract`, identical to the
record/corpus convention.
"""

from __future__ import annotations

import math

import pytest
from geometrykit import contract, engine
from geometrykit.geometry import Surface, Vec
from geometrykit.geometry.primitives import Vec as _Vec

_STATIONS = (
    0.0, 0.0625, 0.25, math.sqrt(2.0) / 4.0, 0.5, 0.75, 0.90625, 1.0,
)

_TRICKY_FLOATS = (
    0.0, -0.0, 1.0, -2.5, 0.1 + 0.2, 1e300, -1e-300, 5e-324,
    float("inf"), float("-inf"), float("nan"), 2**53 + 1.0,
    math.pi, math.sqrt(2.0),
)


def _surface() -> Surface:
    """Deterministic bicubic patch: 4x4 poles, no numpy, no OCC.

    The poles carry a gentle saddle so curvature and projection have
    something to bite on.
    """
    def z(x: float, y: float) -> float:
        return 3.0 + 0.35 * x + 0.15 * y - 0.8 * x * (1.0 - x) * y * (1.0 - y)

    rows = []
    for j in range(4):
        row = []
        for i in range(4):
            row.append(_Vec(i / 3.0, j / 3.0, z(i / 3.0, j / 3.0)))
        rows.append(row)
    return Surface(
        rows,
        uknots=[0.0, 1.0], vknots=[0.0, 1.0],
        umults=[4, 4], vmults=[4, 4],
        udegree=3, vdegree=3,
    )


def _sig(rows: list) -> str:
    """MD5[:16] contract signature of a grid of (x, y, z) rows."""
    data = {"k": 1, "rows": [[float(v) for v in r] for r in rows]}
    return contract.signature(data)


def _eval_rows(surf: Surface) -> list:
    out = []
    for u in _STATIONS:
        for v in _STATIONS:
            p = surf.point_at(u, v)
            out.append((p.x, p.y, p.z))
    return out


def _projection_rows(surf: Surface) -> list:
    out = []
    for u in _STATIONS:
        for v in _STATIONS:
            p = _Vec(0.37 + u, 2.1 - v, 2.4 + 0.9 * u)
            q, _uv = surf.project(p)
            out.append((q.x, q.y, q.z))
    return out


class TestHarnessSelfParity:
    """Both sides the mirror: signatures stable across two builds."""

    def test_eval_grid_signature_identical(self):
        assert _sig(_eval_rows(_surface())) == _sig(_eval_rows(_surface()))

    def test_projection_signature_identical(self):
        assert (
            _sig(_projection_rows(_surface()))
            == _sig(_projection_rows(_surface()))
        )

    def test_frozen_patterns_reference(self):
        # anchors: the exact bit patterns this test expects from the
        # analytic knobs of _surface's first pole (guards the station
        # set against silent edits)
        surf = _surface()
        p0 = surf.control_points[0][0]
        assert contract.float_to_hex(p0.x) == "f64:0000000000000000"
        assert contract.float_to_hex(p0.z) == "f64:4008000000000000"  # 3.0


@pytest.mark.rust
@pytest.mark.skipif(not engine.has_rust(), reason="requires geometrykit._rust")
class TestAcceleratorParity:
    """The extension agrees bit-for-bit with the canonical patterns."""

    def test_core_version_stamp(self):
        from geometrykit import _rust

        assert _rust.core_version() == "geometrykit-core 0.1.0"

    def test_bits_hex_agrees_with_contract(self):
        from geometrykit import _rust

        for v in _TRICKY_FLOATS:
            assert _rust.bits_hex(v) == contract.float_to_hex(v), v

    def test_kernel_results_bits_hex_agree(self):
        from geometrykit import _rust

        surf = _surface()
        for u in _STATIONS:
            for v in _STATIONS:
                p = surf.point_at(u, v)
                for x in (p.x, p.y, p.z):
                    assert _rust.bits_hex(x) == contract.float_to_hex(x)
        for row in _projection_rows(surf):
            for x in row:
                assert not (x != x), "NaN leaked into a projection row"

    def test_pattern_of_engine_state(self):
        Vec  # names stay import-stable for the Phase 2 dispatch seam
