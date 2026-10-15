# This file was generated with the assistance of an AI coding tool.
"""Plane x surface sections with pcurves, all kernel-native.

Plain tier: analytic straight sections, structural guarantees (points on
the plane, pcurve consistency by construction, empty miss), the
CurveOnSurface record shape. Oracle tier: the same kernel fill surface
sectioned both ways -- agreement with `occ_intersect_plane` point sets.
"""

from __future__ import annotations

import pytest
from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Plane, Vec
from geometrykit.geometry.surface import CurveOnSurface, Surface


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


def _vertical_plane(x: float = 4.0) -> Plane:
    return Plane(origin=Vec(x, 0, 0), x_axis=Vec(0, 1, 0), y_axis=Vec(0, 0, 1))


def _network():
    return [
        Curve.from_tangents(
            start=Vec(0, 0, 2.4), tan_start=Vec(1, 0, 0),
            end=Vec(8, 0, 2.8), tan_end=Vec(1, 0, 0), scale=3.0,
        ),
        Curve.from_tangents(
            start=Vec(8, 0, 2.8), tan_start=Vec(0, 1, 0),
            end=Vec(8, 6, 2.9), tan_end=Vec(0, 1, 0), scale=3.0,
        ),
        Curve.from_tangents(
            start=Vec(0, 6, 3.2), tan_start=Vec(1, 0, 0),
            end=Vec(8, 6, 2.9), tan_end=Vec(1, 0, 0), scale=3.0,
        ),
        Curve.from_tangents(
            start=Vec(0, 6, 3.2), tan_start=Vec(0, -1, 0),
            end=Vec(0, 0, 2.4), tan_end=Vec(0, -1, 0), scale=3.0,
        ),
    ]


class TestStraightSection:
    """Analytic: a vertical plane through a flat patch is a straight run."""

    def test_single_straight_chain(self):
        sections = _flat_plane().section(_vertical_plane())
        assert len(sections) == 1
        assert isinstance(sections[0], CurveOnSurface)
        curve = sections[0].curve
        for i in range(21):
            p = curve.point_at(i / 20)
            assert abs(p.x - 4.0) < 1e-9, (i, p)

    def test_pcurve_reproduces_the_chain(self):
        sections = _flat_plane().section(_vertical_plane())
        entry = sections[0]
        surface = _flat_plane()
        for (u, v), point in zip(entry.uv, entry_points(entry)):
            q = surface.point_at(u, v)
            assert abs(q - point) < 1e-12, ((u, v), point)

    def test_miss_returns_empty(self):
        sections = _flat_plane().section(_vertical_plane(x=20.0))
        assert sections == []


def entry_points(entry: CurveOnSurface) -> list:
    from geometrykit.runtimes._npu import surface_np

    out = []
    for u, v in entry.uv:
        q = surface_np.eval_point(entry.surface, u, v)
        out.append(Vec(q.x, q.y, q.z))
    return out


class TestCurvedSection:
    """Structural: points on the plane AND on the surface, via the track."""

    @staticmethod
    def _fill():
        from geometrykit.runtimes._npu import patch_np

        field, _c = patch_np.discrete_plate_fill(
            _network(), nu=12, nv=12, parametrization="arclength"
        )
        return field

    def test_points_on_plane_and_track_consistent(self):
        field = self._fill()
        plane = _vertical_plane()
        sections = field.section(plane)
        assert len(sections) >= 1
        for entry in sections:
            assert isinstance(entry.uv, list) and len(entry.uv) >= 2
            for (u, v) in entry.uv:
                q = field.point_at(u, v)
                assert abs(q.x - 4.0) < 1e-6, (u, v, q)

    def test_curve_follows_the_track(self):
        field = self._fill()
        sections = field.section(_vertical_plane())
        entry = sections[0]
        worst = 0.0
        for k, (u, v) in enumerate(entry.uv):
            t = k / max(len(entry.uv) - 1, 1)
            p = entry.curve.point_at(t)
            q = field.point_at(u, v)
            worst = max(worst, abs(p.x - q.x), abs(p.y - q.y), abs(p.z - q.z))
        # interpolation error only (the track itself is exact)
        assert worst < 5e-2, worst


def _occ_available() -> bool:
    try:
        import OCC.Core.BRep  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _occ_available(), reason="needs pythonocc as the reference")
class TestAgainstOccSection:
    """Agreement with `occ_intersect_plane` point sets on a kernel fill."""

    def test_section_matches_occ(self):
        from geometrykit.geometry.surface import occ_intersect_plane

        field = TestCurvedSection._fill()
        plane = _vertical_plane()
        ours = field.section(plane)
        theirs = occ_intersect_plane(field, plane)
        assert ours and theirs
        ours_pts = [p for entry in ours for p in entry_points(entry)]
        their_pts = []
        for curve in theirs:
            for i in range(101):
                their_pts.append(curve.point_at(i / 100))
        worst = 0.0
        for p in ours_pts:
            best = min(abs(p - q) for q in their_pts)
            worst = max(worst, best)
        assert worst < 5e-2, worst


class TestSectionLevels:
    """One field evaluation, many parallel slices -- self-consistent."""

    @staticmethod
    def _fill():
        from geometrykit.runtimes._npu import patch_np

        field, _c = patch_np.discrete_plate_fill(
            _network(), nu=12, nv=12, parametrization="arclength"
        )
        return field

    def test_levels_match_repeated_sections(self):
        field = self._fill()
        base = Plane(origin=Vec(0, 0, 0), x_axis=Vec(1, 0, 0), y_axis=Vec(0, 0, -1))
        offsets = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
        batches = field.section_levels(base, offsets)
        assert len(batches) == len(offsets)
        for offset, curves in zip(offsets, batches):
            single = field.section(
                Plane(
                    origin=Vec(0, offset, 0),
                    x_axis=Vec(1, 0, 0),
                    y_axis=Vec(0, 0, -1),
                )
            )
            assert len(curves) == len(single), (offset, len(curves), len(single))
            for a, b in zip(curves, single):
                assert len(a.uv) == len(b.uv), offset
                for (ua, va), (ub, vb) in zip(a.uv, b.uv):
                    assert abs(ua - ub) + abs(va - vb) < 1e-9, offset

    def test_miss_gives_empty_list(self):
        field = self._fill()
        base = Plane(origin=Vec(0, 0, 0), x_axis=Vec(1, 0, 0), y_axis=Vec(0, 0, -1))
        batches = field.section_levels(base, [50.0])
        assert batches == [[]]
