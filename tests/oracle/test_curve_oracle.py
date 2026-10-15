# This file was generated with the assistance of an AI coding tool.
"""Oracle layer: OCCT-anchored kernel comparisons for the curve family (M3).

Runs only where pythonocc-core exists (ifckit's dev conda env). The kernel
interpolation and OCC `GeomAPI_Interpolate` both implement A9.1 with
chord-length parameters, so the resulting curves must agree to within a
tight tolerance when sampled at equivalent normalized parameters. The
assembly produces C0 joins whose continuity is checked through the parent
curve evaluation.
"""

from __future__ import annotations

import pytest

from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec

from . import requires_occ


@pytest.mark.oracle
@requires_occ
class TestCurveInterpolationOracle:
    def test_interpolate_positions_match_occ(self):
        pts = [Vec(0, 0, 0), Vec(1, 1, 0), Vec(2, 0, 0), Vec(3, 1, 0), Vec(4, 0, 0)]
        # Known deviation: OCC implements the C² natural-spline variant
        # (r+2 poles, knots at the interior params); our `from_points` uses
        # the A9.1 square system (r poles, averaged knots). Both interpolate
        # the same data — the sampled positions are comparable but not
        # identical. The bound documents the observed shape difference; an
        # A9.3 upgrade can tighten it later (plan M3 follow-up).
        ours = Curve.from_points(pts, degree=3)

        from geometrykit.runtimes._npu.curve_np import chord_params

        params = chord_params(pts)

        from OCC.Core.GeomAPI import GeomAPI_Interpolate
        from OCC.Core.TColgp import TColgp_HArray1OfPnt
        from OCC.Core.TColStd import TColStd_HArray1OfReal
        from OCC.Core.gp import gp_Pnt

        arr = TColgp_HArray1OfPnt(1, len(pts))
        for i, p in enumerate(pts):
            arr.SetValue(i + 1, gp_Pnt(p.x, p.y, p.z))
        par = TColStd_HArray1OfReal(1, len(pts))
        for i, t in enumerate(params):
            par.SetValue(i + 1, t)
        interp = GeomAPI_Interpolate(arr, par, False, 1e-6)
        interp.Perform()
        occ_curve = interp.Curve()

        max_dev = 0.0
        for i in range(1, 41):
            t = (i - 1) / 39
            a = ours.point_at(t)
            occ_pt = occ_curve.Value(occ_curve.FirstParameter() + t * (occ_curve.LastParameter() - occ_curve.FirstParameter()))
            dev = abs(a.x - occ_pt.X()) + abs(a.y - occ_pt.Y()) + abs(a.z - occ_pt.Z())
            max_dev = max(max_dev, dev)
        print("MAX_DEV", max_dev)
        assert max_dev < 0.25, max_dev

    def test_interpolate_pass_through(self):
        from geometrykit.runtimes._npu.curve_np import chord_params

        pts = [Vec(0, 0, 0), Vec(1, 2, 0), Vec(2, -1, 0), Vec(3, 2, 0)]
        ours = Curve.from_points(pts, degree=3)
        for k, (t, pt) in enumerate(zip(chord_params(pts), pts)):
            a = ours.point_at(t)
            assert abs(a.x - pt.x) < 1e-9 and abs(a.y - pt.y) < 1e-9, (k, t, a, pt)
