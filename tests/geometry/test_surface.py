"""
Tests for geometrykit.geometry.surface — Surface (NURBS/BSpline), pure Python parts.
"""

from __future__ import annotations

import pytest
from geometrykit.geometry import Plane, Surface, Transform, Vec

TOL = 1e-6


def _simple_surface() -> Surface:
    return Surface(
        control_points=[
            [Vec(0, 0, 0), Vec(5, 0, 0)],
            [Vec(0, 5, 0), Vec(5, 5, 0)],
            [Vec(0, 10, 1), Vec(5, 10, 1)],
        ],
        uknots=[0.0, 1.0],
        vknots=[0.0, 1.0],
        umults=[3, 3],
        vmults=[2, 2],
        udegree=2,
        vdegree=1,
    )


def _rational_surface() -> Surface:
    return Surface(
        control_points=[
            [Vec(1, 0, 0), Vec(1, 1, 0)],
            [Vec(0, 0, 0), Vec(0, 1, 0)],
            [Vec(-1, 0, 0), Vec(-1, 1, 0)],
        ],
        uknots=[0.0, 0.5, 1.0],
        vknots=[0.0, 1.0],
        umults=[3, 1, 2],
        vmults=[2, 2],
        udegree=2,
        vdegree=1,
        weights=[
            [1.0, 1.0],
            [2.0, 2.0],
            [1.0, 1.0],
        ],
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_simple_surface(self):
        s = _simple_surface()
        assert s.nu == 3
        assert s.nv == 2

    def test_empty_surface(self):
        s = Surface(
            control_points=[],
            uknots=[0.0, 1.0],
            vknots=[0.0, 1.0],
            umults=[2, 2],
            vmults=[2, 2],
            udegree=1,
            vdegree=1,
        )
        assert s.nu == 0
        assert s.nv == 0

    def test_tuples_as_control_points(self):
        s = Surface(
            control_points=[[(0, 0, 0), (1, 0, 0)], [(0, 1, 0), (1, 1, 0)]],
            uknots=[0.0, 1.0],
            vknots=[0.0, 1.0],
            umults=[2, 2],
            vmults=[2, 2],
            udegree=1,
            vdegree=1,
        )
        assert s.nu == 2
        assert s.nv == 2


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_rational(self):
        assert _simple_surface().rational is False
        assert _rational_surface().rational is True

    def test_closed_flags(self):
        s = Surface(
            control_points=[[Vec(0, 0, 0)], [Vec(1, 0, 0)]],
            uknots=[0.0, 1.0],
            vknots=[0.0, 1.0],
            umults=[2, 2],
            vmults=[2, 1],
            udegree=1,
            vdegree=1,
            uclosed=True,
            vclosed=True,
        )
        assert s.uclosed is True
        assert s.vclosed is True


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


class TestTransform:
    def test_translate(self):
        s = _simple_surface()
        r = s.translated(Vec(0, 0, 10))
        assert (r.control_points[0][0] - Vec(0, 0, 10)).length() < TOL

    def test_scale(self):
        s = _simple_surface()
        r = s.scaled(2.0, 2.0, 2.0)
        assert (r.control_points[1][1] - Vec(10, 10, 0)).length() < TOL

    def test_mirror(self):
        plane = Plane(Vec(0, 0, 0), Vec(1, 0, 0), Vec(0, 1, 0))
        s = _simple_surface()
        r = s.mirrored(plane)
        assert (r.control_points[0][1] - Vec(5, 0, 0)).length() < TOL

    def test_rotate(self):
        s = Surface(
            control_points=[[Vec(1, 0, 0)], [Vec(1, 1, 0)]],
            uknots=[0.0, 1.0],
            vknots=[0.0, 1.0],
            umults=[2, 2],
            vmults=[2, 1],
            udegree=1,
            vdegree=1,
        )
        r = s.rotated(Vec(0, 0, 1), 1.57079632679)
        assert r.control_points[0][0].y == pytest.approx(1.0, abs=TOL)

    def test_transformed(self):
        s = _simple_surface()
        t = Transform.translation(Vec(1, 2, 3))
        r = s.transformed(t)
        assert (r.control_points[0][0] - Vec(1, 2, 3)).length() < TOL

    def test_copy(self):
        s = _rational_surface()
        c = s.copy()
        assert c.rational is True
        assert c._weights is not s._weights


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_to_dict_from_dict_roundtrip(self):
        s = _simple_surface()
        d = s.to_dict()
        r = Surface.from_dict(d)
        assert r.nu == s.nu
        assert r.nv == s.nv
        assert r.udegree == s.udegree
        assert r.vdegree == s.vdegree

    def test_rational_roundtrip(self):
        s = _rational_surface()
        d = s.to_dict()
        r = Surface.from_dict(d)
        assert r.rational is True
        assert r.uclosed == s.uclosed
        assert r.vclosed == s.vclosed

    def test_closed_roundtrip(self):
        s = Surface(
            control_points=[[Vec(0, 0, 0)], [Vec(1, 0, 0)]],
            uknots=[0.0, 1.0],
            vknots=[0.0, 1.0],
            umults=[2, 2],
            vmults=[2, 1],
            udegree=1,
            vdegree=1,
            uclosed=True,
            vclosed=False,
        )
        d = s.to_dict()
        r = Surface.from_dict(d)
        assert r.uclosed is True
        assert r.vclosed is False


# ---------------------------------------------------------------------------
# Projection (P0: coarse grid + damped Newton + boundary fallback)
# ---------------------------------------------------------------------------


def _flat_plane() -> Surface:
    return Surface(
        control_points=[
            [Vec(0, 0, 0), Vec(0, 6, 0)],
            [Vec(8, 0, 0), Vec(8, 6, 0)],
        ],
        uknots=[0.0, 1.0],
        vknots=[0.0, 1.0],
        umults=[2, 2],
        vmults=[2, 2],
        udegree=1,
        vdegree=1,
    )


class TestProjectPoint:
    """Analytic anchors: plane exact, outside finds the boundary."""

    def test_interior_projects_exactly(self):
        from geometrykit.runtimes._npu import surface_np

        plane = _flat_plane()
        for pt, want in (
            (Vec(4, 3, 2.5), (4.0, 3.0, 0.0)),
            (Vec(0, 0, 1.0), (0.0, 0.0, 0.0)),
            (Vec(8, 6, 0.5), (8.0, 6.0, 0.0)),
        ):
            q, uv = surface_np.project_point(plane, pt)
            assert max(abs(q.x - want[0]), abs(q.y - want[1]), abs(q.z - want[2])) < 1e-9, (pt, q)
            assert uv[0] == pytest.approx(want[0] / 8.0)
            assert uv[1] == pytest.approx(want[1] / 6.0)

    def test_outside_footprint_finds_the_boundary(self):
        from geometrykit.runtimes._npu import surface_np

        plane = _flat_plane()
        q, _uv = surface_np.project_point(plane, Vec(10, 3, 1.0))
        assert max(abs(q.x - 8), abs(q.y - 3), abs(q.z)) < 1e-9, q
        q, _uv = surface_np.project_point(plane, Vec(4, -2, 0.5))
        assert max(abs(q.x - 4), abs(q.y), abs(q.z)) < 1e-9, q

    def test_tilted_plane_projects_exactly(self):
        from geometrykit.runtimes._npu import surface_np

        tilt = Surface(
            control_points=[
                [Vec(0, 0, 0), Vec(0, 1, 1)],
                [Vec(1, 0, 0), Vec(1, 1, 1)],
            ],
            uknots=[0.0, 1.0],
            vknots=[0.0, 1.0],
            umults=[2, 2],
            vmults=[2, 2],
            udegree=1,
            vdegree=1,
        )
        q, _uv = surface_np.project_point(tilt, Vec(0.5, 0.5, 2.0))
        assert max(abs(q.x - 0.5), abs(q.y - 1.0), abs(q.z - 1.0)) < 1e-9, q

    def test_eval_all_agrees_with_split_kernels(self):
        from geometrykit.runtimes._npu import surface_np

        plane = _flat_plane()
        for u, v in ((0.0, 0.0), (0.3, 0.7), (1.0, 1.0)):
            p, su, sv = surface_np.eval_all(plane, u, v)
            q = surface_np.eval_point(plane, u, v)
            tu, tv = surface_np.eval_tangents(plane, u, v)
            assert (p.x, p.y, p.z) == (q.x, q.y, q.z)
            assert (su.x, su.y, su.z) == (tu.x, tu.y, tu.z)
            assert (sv.x, sv.y, sv.z) == (tv.x, tv.y, tv.z)

    def test_surface_project_memoizes(self):
        plane = _flat_plane()
        assert getattr(plane, "_project_cache", None) is None
        plane.project(Vec(4, 3, 2.5))
        assert len(plane._project_cache) == 1
        q, _uv = plane.project(Vec(4, 3, 2.5))
        assert max(abs(q.x - 4), abs(q.y - 3), abs(q.z)) < 1e-9
