# This file was generated with the assistance of an AI coding tool.
"""Patch kernel tests: P-0 discrete Coons (occ-free) + routing."""

from __future__ import annotations

import math

import pytest

from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec
from geometrykit.geometry.surface import Surface


def _rect_lines(w: float = 8.0, h: float = 4.0, z: float = 0.0) -> list:
    corners = [Vec(0, 0, z), Vec(w, 0, z), Vec(w, h, z), Vec(0, h, z)]
    return [
        Curve(
            control_points=[corners[i], corners[(i + 1) % 4]],
            knots=[0.0, 1.0],
            multiplicities=[2, 2],
            degree=1,
        )
        for i in range(4)
    ]


class TestP0DiscreteCoons:
    def test_rectangular_patch_corners(self):
        s = Surface.patch(_rect_lines())
        c00 = s.point_at(0.0, 0.0)
        c10 = s.point_at(1.0, 0.0)
        c01 = s.point_at(0.0, 1.0)
        c11 = s.point_at(1.0, 1.0)
        for c, exp in ((c00, (0, 0, 0)), (c10, (8, 0, 0)), (c01, (0, 4, 0)), (c11, (8, 4, 0))):
            assert (c.x, c.y, c.z) == pytest.approx(exp), c

    def test_midpoint_center_of_rectangle(self):
        s = Surface.patch(_rect_lines())
        m = s.point_at(0.5, 0.5)
        assert (m.x, m.y, m.z) == pytest.approx((4.0, 2.0, 0.0))

    def test_planarity_coplanar_reflexive_g1(self):
        """Planar boundaries in → planar patch out (the coplanar G1 case)."""
        s = Surface.patch(_rect_lines())
        zs = [s.point_at(i / 10.0, j / 10.0).z for i in range(11) for j in range(11)]
        assert max(abs(z) for z in zs) < 1e-9

    def test_flow_roof_coplanar_patch(self):
        """The flowing-roof default corpus (all heights 2.5) is coplanar — the
        kernel patch must produce the planar sheet (reflexive G1)."""
        w, h = 8.0, 12.0
        corners = [
            Vec(0, 0, 2.5),
            Vec(w, 0, 2.5),
            Vec(w, h / 2, 2.5),
            Vec(0, h / 2, 2.5),
        ]
        curves = [
            Curve(
                control_points=[corners[i], corners[(i + 1) % 4]],
                knots=[0.0, 1.0],
                multiplicities=[2, 2],
                degree=1,
            )
            for i in range(4)
        ]
        s = Surface.patch(curves)  # no constraints → G0 → kernel P-0
        zs = [s.point_at(i / 8.0, j / 8.0).z for i in range(9) for j in range(9)]
        assert max(abs(z - 2.5) for z in zs) < 1e-9

    def test_degree3_boundaries_still_pass_through(self):
        """Curved (degree-3 from_tangents) boundaries: corners pass through."""
        c_south = Curve.from_tangents(
            start=Vec(0, 0, 0), tan_start=Vec(1, 0, 0),
            end=Vec(8, 0, 0), tan_end=Vec(1, 0, 0), scale=3.0,
        )
        c_east = Curve.from_tangents(
            start=Vec(8, 0, 0), tan_start=Vec(0, 1, 0),
            end=Vec(8, 4, 0), tan_end=Vec(0, 1, 0), scale=3.0,
        )
        c_north = Curve.from_tangents(
            start=Vec(0, 4, 0), tan_start=Vec(1, 0, 0),
            end=Vec(8, 4, 0), tan_end=Vec(1, 0, 0), scale=3.0,
        )
        c_west = Curve.from_tangents(
            start=Vec(0, 4, 0), tan_start=Vec(0, -1, 0),
            end=Vec(0, 0, 0), tan_end=Vec(0, -1, 0), scale=3.0,
        )
        s = Surface.patch([c_south, c_east, c_north, c_west])
        assert (s.point_at(0.0, 0.0).x, s.point_at(0.0, 0.0).y) == pytest.approx((0, 0))
        assert (s.point_at(1.0, 1.0).x, s.point_at(1.0, 1.0).y) == pytest.approx((8, 4))

    def test_corner_mismatch_raises(self):
        """A gap in the corner cycle raises (G0 validated before solving)."""
        c_south = Curve(
            control_points=[Vec(0, 0, 0), Vec(4, 0, 0)],
            knots=[0, 1], multiplicities=[2, 2], degree=1,
        )  # ends at (4,0) but nothing continues from there
        c_east = Curve(
            control_points=[Vec(8, 0, 0), Vec(8, 4, 0)],
            knots=[0, 1], multiplicities=[2, 2], degree=1,
        )
        c_north = Curve(
            control_points=[Vec(0, 4, 0), Vec(8, 4, 0)],
            knots=[0, 1], multiplicities=[2, 2], degree=1,
        )
        c_west = Curve(
            control_points=[Vec(0, 4, 0), Vec(0, 0, 0)],
            knots=[0, 1], multiplicities=[2, 2], degree=1,
        )
        with pytest.raises(Exception):
            Surface.patch([c_south, c_east, c_north, c_west])
