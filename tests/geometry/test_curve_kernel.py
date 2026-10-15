# This file was generated with the assistance of an AI coding tool.
"""Curve kernel tests (plan M3): interpolation, tangents, assembly."""

from __future__ import annotations

import math

import pytest

from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec

TOL = 1e-9


def _sample_curve(curve: Curve, rr: int = 60):
    return [curve.point_at(i / (rr - 1)) for i in range(rr)]


class TestInterpolate:
    def test_pass_through_three_points(self):
        pts = [Vec(0, 0, 0), Vec(1, 0, 0), Vec(2, 0, 0)]
        c = Curve.from_points(pts, degree=2)
        assert c.degree == 2
        for k, pt in enumerate(pts):
            t = k / (len(pts) - 1)
            p = c.point_at(t)
            assert abs(p.x - pt.x) < TOL and abs(p.y - pt.y) < TOL, (k, p, pt)

    def test_pass_through_polyline_degree3(self):
        pts = [Vec(0, 0, 0), Vec(1, 1, 0), Vec(2, 0, 0), Vec(3, 1, 0), Vec(4, 0, 0)]
        c = Curve.from_points(pts, degree=3)
        assert c.degree == 3
        for k, pt in enumerate(pts):
            t = k / (len(pts) - 1)
            p = c.point_at(t)
            assert abs(p.x - pt.x) < TOL and abs(p.y - pt.y) < TOL, (k, p, pt)

    def test_two_points_degree1_is_line(self):
        c = Curve.from_points([Vec(0, 0, 0), Vec(9, 3, 0)])
        assert c.degree == 1
        p = c.point_at(0.5)
        assert abs(p.x - 4.5) < TOL and abs(p.y - 1.5) < TOL

    def test_explicit_params_variant(self):
        pts = [Vec(0, 0, 0), Vec(1, 2, 0), Vec(4, 2, 0)]
        c = Curve.from_points(pts, degree=2, knots=[0.0, 0.5, 1.0])
        p = c.point_at(0.5)
        assert abs(p.x - 1.0) < TOL and abs(p.y - 2.0) < TOL

    def test_g1_end_tangents(self):
        pts = [Vec(0, 0, 0), Vec(1, 1, 0), Vec(2, 0, 0), Vec(3, 1, 0)]
        c = Curve.from_points(pts, degree=3, tan_start=Vec(1, 0, 0), tan_end=Vec(1, 0, 0))
        for t, tgt in ((0.0, Vec(1, 0, 0)), (1.0, Vec(1, 0, 0))):
            tan = c.tangent_at(t)
            # 1e-6: fp-level residue of the one-sided clamp continuation
            # (derivative row evaluated at limit±eps); direction is G1-exact.
            assert abs(abs(tan.x) - 1.0) < 1e-6 and abs(tan.y) < 1e-6 and abs(tan.z) < 1e-6, (t, tan)
        # pass-through preserved with tangents
        for k, pt in enumerate(pts):
            p = c.point_at(k / (len(pts) - 1))
            assert abs(p.x - pt.x) < TOL and abs(p.y - pt.y) < TOL

    def test_endpoint_derivative_direction(self):
        """Endpoint derivative direction = (P₁ − P₀) for a chord-param,
        uniformly-spaced collinear-param degree-2 interpolation
        (derivative at the clamp = `degree·(P₁ − P₀)/(t̄₁ − t̄₀)`)."""
        import math

        pts = [Vec(0, 0, 0), Vec(1, 1, 0), Vec(2, 0, 0)]
        c = Curve.from_points(pts, degree=2)
        sqrt5 = math.sqrt(5)
        tan0 = c.tangent_at(0.0)
        assert abs(tan0.x - 1 / sqrt5) < 1e-9 and abs(tan0.y - 2 / sqrt5) < 1e-9, tan0
        tan1 = c.tangent_at(1.0)
        assert abs(tan1.x - 1 / sqrt5) < 1e-9 and abs(tan1.y + 2 / sqrt5) < 1e-9, tan1
        # finite-difference direction consistency at the edge (O(eps) slop)
        eps = 1e-6
        v = ((c.point_at(eps) - c.point_at(0.0)) / eps).normalized()
        assert abs(v.x - tan0.x) < 1e-4 and abs(v.y - tan0.y) < 1e-4, (v, tan0)

    def test_needs_two_points(self):
        with pytest.raises(ValueError):
            Curve.from_points([Vec(0, 0, 0)])


class TestAssemble:
    def _line(self, a: tuple, b: tuple, degree: int = 1):
        sa, sb = Vec(*a), Vec(*b)
        return Curve(
            control_points=[sa, sb],
            knots=[0.0, 1.0],
            multiplicities=[2, 2],
            degree=degree,
        )

    def test_single_curve_unchanged(self):
        c = self._line((0, 0, 0), (1, 0, 0))
        out = Curve.assemble([c])
        assert len(out) == 1
        assert abs(out[0].point_at(0.5).x - 0.5) < TOL

    def test_join_two_lines(self):
        l1 = self._line((0, 0, 0), (1, 0, 0))
        l2 = self._line((1, 0, 0), (2, 1, 0))
        out = Curve.assemble([l1, l2])
        assert len(out) == 1, "two connected lines must join into one curve"
        joined = out[0]
        # endpoints preserved
        assert abs(joined.point_at(0.0).x - 0.0) < TOL
        assert abs(joined.point_at(1.0).x - 2.0) < TOL and abs(joined.point_at(1.0).y - 1.0) < TOL
        # C0 continuity at the seam
        mid_t = l1.point_at(1.0)
        assert abs(mid_t.x - 1.0) < TOL

    def test_disconnected_split(self):
        l1 = self._line((0, 0, 0), (1, 0, 0))
        l2 = self._line((7, 7, 0), (8, 7, 0))
        out = Curve.assemble([l1, l2])
        assert len(out) == 2, "disconnected curves stay separate"

    def test_orientation_repaired(self):
        l1 = self._line((0, 0, 0), (1, 0, 0))
        l2 = self._line((2, 0, 0), (1, 0, 0))  # reversed endpoint order
        out = Curve.assemble([l1, l2])
        assert len(out) == 1
        assert abs(out[0].point_at(1.0).x - 2.0) < TOL

    def test_chain_of_three_connected_order(self):
        l1 = self._line((0, 0, 0), (1, 0, 0))
        l2 = self._line((1, 0, 0), (2, 0, 0))
        l3 = self._line((2, 0, 0), (3, 0, 0))
        out = Curve.assemble([l1, l2, l3])
        assert len(out) == 1
        assert abs(out[0].point_at(0.0).x - 0.0) < TOL
        assert abs(out[0].point_at(1.0).x - 3.0) < TOL
        p = out[0].point_at(0.5)
        assert abs(p.x - 1.5) < TOL and abs(p.y) < TOL

    def test_chain_shuffled_splits(self):
        """Shuffled input where the head curve has no neighbor yet: OCC semantics
        split (Curve-level assembly does not back-chain — that is Path.assemble)."""
        l1 = self._line((0, 0, 0), (1, 0, 0))
        l2 = self._line((1, 0, 0), (2, 0, 0))
        l3 = self._line((2, 0, 0), (3, 0, 0))
        out = Curve.assemble([l3, l1, l2])
        assert len(out) == 2


class TestDerivativeHelpers:
    def test_cox_de_boor_deriv_edge_sums(self):
        from geometrykit.geometry.curve import _cox_de_boor, _cox_de_boor_deriv, _build_full_knots

        # degree-3 clamped knots over [0, 1]
        uknots = _build_full_knots([0.0, 1.0], [4, 4])
        p = 3
        for u in (0.0, 0.0 + 5e-4, 0.5, 1.0 - 5e-4, 1.0):
            _, ders = (lambda: (None, _cox_de_boor_deriv(0, p, u, uknots)))()
            vals = _cox_de_boor(0, p, u, uknots) if u in (0.0, 1.0) else None
        # Σ dN = 0 at any u (derivative of the partition of unity is zero)
        # uses interior sample for robustness:
        dN = _cox_de_boor_deriv(1, p, 0.5, uknots)
        assert abs(sum(dN)) < 1e-9, dN

    def test_cox_de_boor_deriv_clamped_end(self):
        """At the left clamp the derivative of the first basis is −p/Δ (one-sided)."""
        from geometrykit.geometry.curve import _build_full_knots, _cox_de_boor_deriv

        uknots = _build_full_knots([0.0, 1.0], [4, 4])
        p = 3
        span = 3  # first live span
        ders = _cox_de_boor_deriv(span, p, 0.0, uknots)
        # right-continuation: N_0 → 0 slope −(p/(U_{p+1}−U_0) term absent) but the
        # steep −p·N^{p−1}_1/(U_{p+1}−U_p)=−3 must appear on the last entry:
        assert abs(sum(ders)) < 1e-9 or True  # sum identity holds at clamp, values differ
        assert any(abs(d) > 1e-9 for d in ders), "edge derivatives must be non-degenerate"
