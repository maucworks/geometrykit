# This file was generated with the assistance of an AI coding tool.
"""Pure-Python ribbons: exact rails, exact spine placement, no OCC.

The ribbon used to be ruled by OCCT's BRepFill_Generator; now the rule
is exact tensor arithmetic over two same-parameterised rails (the spine
itself one-sided, both rails offset control nets symmetric), so the
spine sits *exactly* on the edge / middle line instead of approximately.

Two tiers, as usual: structural properties (spine placement, width)
run everywhere; agreement with the OCC-ruled surface is oracle-gated,
with the OCC generator inlined as the live reference (twenty lines --
its job is being the reference, and it tracks the OCC version).
"""

from __future__ import annotations

import math

import pytest
from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec
from geometrykit.geometry.surface import Surface


def _spine() -> Curve:
    return Curve.from_tangents(
        start=Vec(0, 0, 2.5), tan_start=Vec(1, 0, 0),
        end=Vec(8, 0, 2.5), tan_end=Vec(1, 0, 0), scale=3.0,
    )


def _occ_available() -> bool:
    try:
        import OCC.Core.BRep  # noqa: F401
        return True
    except ImportError:
        return False


def _occ_ruled(rail_a, rail_b):
    """The old construction, live: OCC-ruled between two kernel rails.

    Rails go through `_build_occ_edge`, exactly as the library did before
    the pure rewrite -- the reference is the deleted code path itself.
    """
    from geometrykit.geometry.surface import _build_occ_edge
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
    from OCC.Core.BRepFill import BRepFill_Generator
    from OCC.Core.Geom import Geom_BSplineSurface
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer

    gen = BRepFill_Generator()
    gen.AddWire(BRepBuilderAPI_MakeWire(_build_occ_edge(rail_a)).Wire())
    gen.AddWire(BRepBuilderAPI_MakeWire(_build_occ_edge(rail_b)).Wire())
    gen.Perform()
    exp = TopExp_Explorer(gen.Shell(), TopAbs_FACE)
    assert exp.More()
    from OCC.Core.BRep import BRep_Tool

    bspline = Geom_BSplineSurface.DownCast(BRep_Tool.Surface(exp.Current()))
    assert bspline is not None
    return Surface.from_occ_surface(bspline)


class TestPureRibbon:
    """Structure: where the spine sits, how wide the strip is."""

    @pytest.mark.parametrize("symmetric", [False, True])
    def test_spine_sits_exactly(self, symmetric):
        spine = _spine()
        ribbon = Surface.ribbon(spine, angle_deg=0, width=1.0, symmetric=symmetric)
        v = 0.5 if symmetric else 0.0
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert abs(ribbon.point_at(t, v) - spine.point_at(t)) < 1e-12, t

    def test_width_is_exact_midspan(self):
        spine = _spine()
        for symmetric in (False, True):
            ribbon = Surface.ribbon(spine, angle_deg=0, width=1.0, symmetric=symmetric)
            assert abs(abs(ribbon.point_at(0.5, 0.0) - ribbon.point_at(0.5, 1.0)) - 1.0) < 1e-9

    def test_twisted_ribbon_runs_plain(self):
        spine = _spine()
        ribbon = Surface.ribbon(
            spine, angles_deg=[180, 155, 180], width=1.0, symmetric=True
        )
        assert ribbon.nu > 0 and ribbon.nv == 2
        for t in (0.0, 0.5, 1.0):
            assert abs(ribbon.point_at(t, 0.5) - spine.point_at(t)) < 1e-12, t

    def test_degree_one_across(self):
        ribbon = Surface.ribbon(_spine(), angle_deg=0, width=1.0, symmetric=True)
        assert ribbon.vdegree == 1
        assert ribbon.vknots == [0.0, 1.0] and ribbon.vmults == [2, 2]


@pytest.mark.skipif(not _occ_available(), reason="needs pythonocc as the reference")
class TestAgainstOccRuling:
    """The pure rule reproduces the OCC-ruled surface where it matters."""

    def test_symmetric_matches_bitwise(self):
        from geometrykit.geometry.surface import _offset_control_net

        spine = _spine()
        up = Vec(0, 0, 1)

        def _perp_at(t):
            from geometrykit.geometry.primitives import Vec as _Vec

            tan = spine.tangent_at(t)
            ln = math.sqrt(tan.x**2 + tan.y**2 + tan.z**2)
            return _Vec(
                (tan.y * up.z - tan.z * up.y) / ln,
                (tan.z * up.x - tan.x * up.z) / ln,
                (tan.x * up.y - tan.y * up.x) / ln,
            )

        rails = [
            _offset_control_net(spine, d, _perp_at) for d in (-0.5, 0.5)
        ]
        occ = _occ_ruled(*rails)
        pure = Surface.ribbon(spine, angle_deg=0, width=1.0, symmetric=True)
        worst = 0.0
        for i in range(21):
            for j in range(5):
                p = pure.point_at(i / 20, j / 4)
                q = occ.point_at(i / 20, j / 4)
                worst = max(worst, abs(p - q))
        assert worst < 1e-9, worst

    def test_one_sided_agrees_at_the_spine(self):

        spine = _spine()
        up = Vec(0, 0, 1)
        far = []
        for i in range(20):
            t = i / 19
            p = spine.point_at(t)
            tan = spine.tangent_at(t)
            ln = math.sqrt(tan.x**2 + tan.y**2 + tan.z**2)
            far.append(
                Vec(
                    p.x + (tan.y * up.z - tan.z * up.y) / ln,
                    p.y + (tan.z * up.x - tan.x * up.z) / ln,
                    p.z + (tan.x * up.y - tan.y * up.x) / ln,
                )
            )
        from geometrykit.geometry.curve import Curve as _Curve

        occ = _occ_ruled(spine, _Curve.from_points(far))
        pure = Surface.ribbon(spine, angle_deg=0, width=1.0)
        worst_edge, worst_plane = 0.0, 0.0
        for i in range(21):
            u = i / 20
            worst_edge = max(worst_edge, abs(pure.point_at(u, 0.0) - occ.point_at(u, 0.0)))
            n = pure.normal_at(u, 0.0)
            m = occ.normal_at(u, 0.0)
            dot = max(-1.0, min(1.0, n.x * m.x + n.y * m.y + n.z * m.z))
            worst_plane = max(worst_plane, math.degrees(math.acos(dot)))
        assert worst_edge < 1e-9, worst_edge
        assert worst_plane < 0.1, worst_plane
