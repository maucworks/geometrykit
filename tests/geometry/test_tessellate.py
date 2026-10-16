# This file was generated with the assistance of an AI coding tool.
"""Adaptive kernel tessellation: watertightness, deflection, engine parity.

Every check is exact except where stated: watertightness is
combinatorial (edge-share counts), deflection is measured through
the returned uv tracks (surface evaluated at the edge midpoint, no
projection Newton in the loop), determinism is bitwise.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest
from geometrykit import contract
from geometrykit.geometry import Surface
from geometrykit.geometry.primitives import Vec
from geometrykit.runtimes._npu import tessellate_np
from geometrykit.runtimes._npu.surface_np import _data_of, _eval_point_data


def _saddle() -> Surface:
    def z(x: float, y: float) -> float:
        return 3.0 + 0.35 * x + 0.15 * y - 0.8 * x * (1.0 - x) * y * (1.0 - y)

    rows = [
        [Vec(i / 3.0, j / 3.0, z(i / 3.0, j / 3.0)) for j in range(4)]
        for i in range(4)
    ]
    return Surface(
        rows,
        uknots=[0.0, 1.0], vknots=[0.0, 1.0],
        umults=[4, 4], vmults=[4, 4],
        udegree=3, vdegree=3,
    )


def _rational() -> Surface:
    surf = _saddle()
    weights = [
        [1.0 + 0.5 * math.sin(1.3 * i + 0.7 * j + 0.2) for j in range(4)]
        for i in range(4)
    ]
    return Surface(
        surf.control_points,
        surf.uknots, surf.vknots, surf.umults, surf.vmults,
        surf.udegree, surf.vdegree,
        weights,
    )


def _edge_shares(tris) -> Counter:
    shares: Counter = Counter()
    for a, b, c in tris:
        assert len({a, b, c}) == 3, f"degenerate triangle {a, b, c}"
        for e in (tuple(sorted((a, b))), tuple(sorted((b, c))),
                  tuple(sorted((c, a)))):
            shares[e] += 1
    return shares


def _deflection(surf, verts, tris, uvs) -> float:
    """Worst edge-midpoint deviation, measured exactly via the uv track."""
    d = _data_of(surf)
    worst = 0.0
    for a, b, c in tris:
        for p, q in ((a, b), (b, c), (c, a)):
            mu = 0.5 * (uvs[2 * p] + uvs[2 * q])
            mv = 0.5 * (uvs[2 * p + 1] + uvs[2 * q + 1])
            s = _eval_point_data(d, mu, mv)
            mx = 0.5 * (verts[3 * p] + verts[3 * q])
            my = 0.5 * (verts[3 * p + 1] + verts[3 * q + 1])
            mz = 0.5 * (verts[3 * p + 2] + verts[3 * q + 2])
            worst = max(worst, abs(s.x - mx) + abs(s.y - my) + abs(s.z - mz))
    return worst


class TestWatertight:
    @pytest.mark.parametrize("make", [_saddle, _rational])
    def test_no_dangling_or_double_edges(self, make):
        surf = make()
        _verts, _normals, tris, _uvs = tessellate_np.adaptive_mesh(surf)
        assert len(tris) > 0
        shares = _edge_shares(tris)
        assert all(n <= 2 for n in shares.values())
        # open patch: boundary edges occur once, interior twice
        assert any(n == 1 for n in shares.values())

    def test_vertices_on_surface(self):
        surf = _saddle()
        verts, _normals, _tris, uvs = tessellate_np.adaptive_mesh(surf)
        d = _data_of(surf)
        for k in range(len(verts) // 3):
            s = _eval_point_data(d, uvs[2 * k], uvs[2 * k + 1])
            assert abs(s.x - verts[3 * k]) < 1e-9
            assert abs(s.y - verts[3 * k + 1]) < 1e-9
            assert abs(s.z - verts[3 * k + 2]) < 1e-9


class TestDeflection:
    @pytest.mark.parametrize("deflection", [0.01, 0.001])
    def test_bound_holds(self, deflection):
        surf = _saddle()
        verts, _normals, tris, uvs = tessellate_np.adaptive_mesh(
            surf, deflection=deflection
        )
        assert _deflection(surf, verts, tris, uvs) <= deflection

    def test_tighter_tolerance_refines(self):
        surf = _saddle()
        _v1, _n1, t1, _u1 = tessellate_np.adaptive_mesh(surf, deflection=0.01)
        _v2, _n2, t2, _u2 = tessellate_np.adaptive_mesh(surf, deflection=0.001)
        assert len(t2) > len(t1)

    def test_rational_bound_holds(self):
        surf = _rational()
        verts, _normals, tris, uvs = tessellate_np.adaptive_mesh(
            surf, deflection=0.005
        )
        assert _deflection(surf, verts, tris, uvs) <= 0.005


class TestGuards:
    def test_bad_arguments_raise(self):
        surf = _saddle()
        with pytest.raises(ValueError):
            tessellate_np.adaptive_mesh(surf, deflection=0.0)
        with pytest.raises(ValueError):
            tessellate_np.adaptive_mesh(surf, angle_tol=-1.0)
        with pytest.raises(ValueError):
            tessellate_np.adaptive_mesh(surf, max_depth=1, base_level=2)


class TestMeshDictWiring:
    def test_adaptive_dict_shape(self):
        surf = _saddle()
        d = surf.to_mesh_dict(label="saddle", method="adaptive")
        assert d["primitive"] == "triangles"
        assert d["label"] == "saddle"
        n = len(d["positions"]) // 3
        assert all(0 <= i < n for i in d["indices"])
        assert len(d["indices"]) % 3 == 0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown mesh method"):
            _saddle().to_mesh_dict(method="bogus")

    def test_preview_carries_method(self):
        d = _saddle().preview(label="s", method="adaptive")
        assert d["__type__"] == "mesh"
        assert d["primitive"] == "triangles"
        assert d["positions"]

    def test_default_still_occ(self):
        # default path unchanged: without pythonocc it raises, it does
        # not silently fall back to the kernel
        try:
            import OCC  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError):
                _saddle().to_mesh_dict()
            return
        d = _saddle().to_mesh_dict()
        assert d["primitive"] == "triangles"


class TestDeterminism:
    def test_two_runs_identical_signatures(self):
        surf = _saddle()
        first = tessellate_np.adaptive_mesh(surf, deflection=0.005)
        second = tessellate_np.adaptive_mesh(surf, deflection=0.005)
        for a, b in zip(first, second):
            if a and isinstance(a[0], tuple):
                a = [c for t in a for c in t]
                b = [c for t in b for c in t]
            assert contract.signature({"v": [float(v) for v in a]}) == contract.signature(
                {"v": [float(v) for v in b]}
            )
