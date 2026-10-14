# This file was generated with the assistance of an AI coding tool.
"""Tests for the flowing-roof standalone stress case.

Role
----
The flowing-roof geometry is the library's stress case and the fixture
set for the remaining OCC-exit milestones:

- Stage 1: seven curves from tangents — pure kernel (M3), no OCC;
- Stage 2: ribbons — OCCT twin until M6; the loud gate is asserted in
  the plain environment and the full build runs in the oracle env;
- Stage 3: two G1-constrained patches on the shared ribbons — OCCT
  twin until M7/M8; the shared `top` ribbon enforces G1 between the
  patches by construction (same support object);
- Stage 4: plane sections + assembly — OCCT twin until M5.

Structural note: the miter planes are the beam-cutting datums — every
edge-curve endpoint lies exactly on its miter plane (pure math).
"""

from __future__ import annotations

import pytest

from examples import flowing_roof
from examples.flowing_roof import (
    RoofGeometry,
    build_curves,
    build_ribbons,
    build_patches,
    build_roof,
    sweep_curves,
)
from geometrykit.geometry.primitives import Vec


def _occ_available() -> bool:
    try:
        import OCC.Core.BRep  # noqa: F401

        return True
    except ImportError:
        return False


DEFAULTS = {
    "num_I_beams": 10,
    "num_L_beams": 50,
    "height_sw": 2.5,
    "height_se": 2.5,
    "height_nw": 2.5,
    "height_ne": 2.5,
    "height_nnw": 2.5,
    "height_nne": 2.5,
    "twist": 25,
}


class TestStage1Curves:
    """Stage 1 (pure kernel, no OCC): the seven edge curves."""

    def test_seven_edge_curves_build(self):
        g = build_curves(DEFAULTS)
        for name in ("bottom", "right", "top", "left", "right2", "top2", "left2"):
            c = getattr(g, name)
            assert c.degree == 3
            assert c.point_at(0.0) is not None

    def test_pass_through_endpoints(self):
        g = build_curves(DEFAULTS)
        assert (g.bottom.point_at(0.0).x, g.bottom.point_at(0.0).y, g.bottom.point_at(0.0).z) == pytest.approx((0.0, 0.0, 2.5))
        assert (g.bottom.point_at(1.0).x, g.bottom.point_at(1.0).y, g.bottom.point_at(1.0).z) == pytest.approx((8.0, 0.0, 2.5))
        assert (g.top.point_at(0.0).x, g.top.point_at(0.0).y, g.top.point_at(0.0).z) == pytest.approx((0.0, 6.0, 2.5))
        assert (g.top2.point_at(1.0).x, g.top2.point_at(1.0).y, g.top2.point_at(1.0).z) == pytest.approx((8.0, 12.0, 2.5))

    def test_shared_edge_joins_patches(self):
        """`top` is shared between both patches: its endpoints anchor the seam."""
        g = build_curves(DEFAULTS)
        for axis in "xyz":
            a = getattr(g.top.point_at(0.0), axis)
            b = getattr(g.left.point_at(0.0), axis)
            assert abs(a - b) < 1e-9, axis
            c = getattr(g.top.point_at(1.0), axis)
            d = getattr(g.right.point_at(1.0), axis)
            assert abs(c - d) < 1e-9, axis

    def test_curve_endpoints_on_miter_planes(self):
        """Edge-curve endpoints lie on their beam-cutting miter planes (pure math)."""
        g = build_curves(DEFAULTS)
        m = g.miters

        def _dist(p, origin, normal):
            o = origin
            nx, ny, nz = normal
            return abs((p.x - o[0]) * nx + (p.y - o[1]) * ny + (p.z - o[2]) * nz)

        # the plane normals are unnormalized directions in these calls; the
        # zero-check is scale-invariant as long as |n| > 0:
        p = g.bottom.point_at(0.0)
        assert _dist(p, (0, 0, 0), (1, -1, 0)) < 1e-9
        p = g.bottom.point_at(1.0)
        assert _dist(p, (8, 0, 0), (1, -1, 0)) < 1e-9
        p = g.right2.point_at(0.0)
        assert _dist(p, (8, 12, 0), (-1, -1, 0)) < 1e-9
        p = g.left2.point_at(1.0)
        assert _dist(p, (0, 12, 0), (-1, 1, 0)) < 1e-9


class TestStage2And3Occt:
    """Ribbon + patch machinery: OCCT blocks until M6-M8 kernels land."""

    @pytest.mark.skipif(not _occ_available(), reason="requires pythonocc-core (OCCT twin)")
    def test_ribbons_and_patches_build(self):
        g = build_roof(DEFAULTS)
        assert len(g.ribbons) == 7
        assert g.surf is not None and g.surf2 is not None
        p = g.surf.point_at(0.5, 0.5)
        assert p is not None

    @pytest.mark.skipif(not _occ_available(), reason="requires pythonocc-core (OCCT twin)")
    def test_sections_and_assembly(self):
        g = build_roof(DEFAULTS)
        sets = sweep_curves(g.surf, g.surf2, Vec(1, 1, 0), num=6, span=30.0)
        assert len(sets) == 6
        non_empty = [curves for _plane, curves in sets if curves]
        assert non_empty, "planes crossing the roof must produce sections"
        for curves in non_empty:
            for c in curves:
                assert c.point_at(0.0) is not None and c.point_at(1.0) is not None

    def test_ribbon_is_loud_without_occ(self):
        """Without pythonocc the ribbon path raises loudly (documented gate)."""
        if _occ_available():
            pytest.skip("OCC present — gate state not testable")
        from geometrykit.geometry.surface import Surface

        g = build_curves(DEFAULTS)
        with pytest.raises(ImportError, match="required"):
            build_ribbons(g)
