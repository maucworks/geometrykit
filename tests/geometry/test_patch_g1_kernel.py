# This file was generated with the assistance of an AI coding tool.
"""P-1 G1 patch tests: coplanar demo (occ-free) + the flowing-roof fixture.

Acceptance is analytic: the patch's cross-derivative at each interior
station must be orthogonal to the constraint surface's normal there
(``tangents_at`` -- no finite differences, no fitting).
"""

from __future__ import annotations

import math

import pytest

from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec
from geometrykit.runtimes._npu import patch_g1, patch_np

UP = (0.0, 0.0, 1.0)


def _occ_available() -> bool:
    try:
        import OCC.Core.BRep  # noqa: F401
        return True
    except ImportError:
        return False


def _rectangle(heights=(2.5, 2.5, 2.5, 2.5)):
    """A four-edge network: south, east, north, west with corner heights
    ``(sw, se, nw, ne)`` — the flowing-roof footprint, parameters free."""
    h_sw, h_se, h_nw, h_ne = heights
    w = 8.0
    d = 6.0
    return [
        Curve.from_tangents(
            start=Vec(0, 0, h_sw), tan_start=Vec(1, 0, 0),
            end=Vec(w, 0, h_se), tan_end=Vec(1, 0, 0), scale=3.0,
        ),
        Curve.from_tangents(
            start=Vec(w, 0, h_se), tan_start=Vec(0, 1, 0),
            end=Vec(w, d, h_ne), tan_end=Vec(0, 1, 0), scale=3.0,
        ),
        Curve.from_tangents(
            start=Vec(0, d, h_nw), tan_start=Vec(1, 0, 0),
            end=Vec(w, d, h_ne), tan_end=Vec(1, 0, 0), scale=3.0,
        ),
        Curve.from_tangents(
            start=Vec(0, d, h_nw), tan_start=Vec(0, -1, 0),
            end=Vec(0, 0, h_sw), tan_end=Vec(0, -1, 0), scale=3.0,
        ),
    ]


def _g1_residual(surface, edge, params, normal_at, band: int = 1) -> float:
    """Worst |cross-derivative . constraint normal| over the enforced stations.

    ``band`` mirrors the solver's corner band (the stations it does not
    constrain). The tolerance lives in the caller: the constraint is exact,
    the 1e-6 band is the evaluation noise between the closed-form boundary
    derivative and the surface evaluator.
    """
    worst = 0.0
    for t in params[1 + band : len(params) - 1 - band]:
        if edge == "south":
            _su, sv = surface.tangents_at(t, 0.0)
            cross = sv
        elif edge == "north":
            _su, sv = surface.tangents_at(t, 1.0)
            cross = sv
        elif edge == "west":
            su, _sv = surface.tangents_at(0.0, t)
            cross = su
        else:
            su, _sv = surface.tangents_at(1.0, t)
            cross = su
        n = normal_at(t)
        worst = max(worst, abs(cross.x * n[0] + cross.y * n[1] + cross.z * n[2]))
    return worst


class TestCoplanarG1:
    """Plain env: the coplanar case -- an already-satisfying field is free."""

    def test_coplanar_g1_leaves_the_patch_untouched(self):
        curves = _rectangle()
        fields = {edge: (lambda t: UP) for edge in patch_g1.EDGE_NAMES}
        base, _corners, stations = patch_g1.constrained_coons_patch(curves, {}, nu=12, nv=12)
        fixed, _corners, _stations = patch_g1.constrained_coons_patch(
            curves, fields, nu=12, nv=12
        )
        moved = max(
            abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)
            for row_a, row_b in zip(base.control_points, fixed.control_points)
            for a, b in zip(row_a, row_b)
        )
        assert moved < 1e-9
        for u in [i / 8 for i in range(9)]:
            for v in [i / 8 for i in range(9)]:
                assert abs(fixed.point_at(u, v).z - 2.5) < 1e-9
        assert stations[0][0] == 0.0 and stations[0][-1] == 1.0

    def test_tilted_field_is_satisfied_exactly(self):
        """A tilted south field: the constraint holds where it was asked to."""
        curves = _rectangle()
        tilt = 0.2
        field = (0.0, -tilt, math.sqrt(1.0 - tilt * tilt))
        fields = {edge: (lambda t: UP) for edge in patch_g1.EDGE_NAMES}
        fields["south"] = lambda t: field
        base, _c, _s = patch_g1.constrained_coons_patch(curves, {}, nu=12, nv=12)
        fixed, _c, stations = patch_g1.constrained_coons_patch(curves, fields, nu=12, nv=12)
        residual = _g1_residual(fixed, "south", stations[0], lambda t: field)
        assert residual < 1e-6, residual
        # the tilt pulls the surface toward the south field, so the patch
        # climbs less steeply off the south edge than the plain Coons base
        _su, sv_base = base.tangents_at(0.5, 0.0)
        _su2, sv_fixed = fixed.tangents_at(0.5, 0.0)
        assert sv_fixed.y < sv_base.y

    def test_field_pairs_and_plain_normals_agree(self):
        curves = _rectangle()
        tilt = 0.2
        field = (0.0, -tilt, math.sqrt(1.0 - tilt * tilt))
        base_fields = {edge: (lambda t: UP) for edge in patch_g1.EDGE_NAMES}
        by_call, _c, stations = patch_g1.constrained_coons_patch(
            curves, dict(base_fields, south=lambda t: field), nu=12, nv=12
        )
        pairs = list(zip(stations[0][1:-1], [field] * (len(stations[0]) - 2)))
        by_pairs, _c, _s = patch_g1.constrained_coons_patch(
            curves, dict(base_fields, south=pairs), nu=12, nv=12
        )
        plain = [field] * (len(stations[0]) - 2)
        by_plain, _c, _s = patch_g1.constrained_coons_patch(
            curves, dict(base_fields, south=plain), nu=12, nv=12
        )
        for a, b, c in zip(by_call.control_points, by_pairs.control_points, by_plain.control_points):
            for pa, pb, pc in zip(a, b, c):
                assert abs(pa.x - pb.x) < 1e-9 and abs(pa.y - pb.y) < 1e-9
                assert abs(pa.x - pc.x) < 1e-9 and abs(pa.y - pc.y) < 1e-9

    def test_unknown_field_raises(self):
        curves = _rectangle()
        with pytest.raises(patch_g1.G1Error):
            patch_g1.constrained_coons_patch(
                curves, {"south": [(0.0, 0.0, 1.0)]}, nu=12, nv=12
            )
        with pytest.raises(patch_g1.G1Error):
            patch_g1.constrained_coons_patch(
                curves, {"south": (0.0, 0.0, 0.0)}, nu=12, nv=12
            )


class TestFlowingRoofG1:
    """The ported flowing-roof fixture with OCCT-sampled ribbon normals."""

    @staticmethod
    def _network():
        w = 8.0
        d = 6.0
        south = Curve.from_tangents(
            start=Vec(0, 0, 2.0), tan_start=Vec(1, 0, 0),
            end=Vec(w, 0, 3.0), tan_end=Vec(1, 0, 0), scale=3.0,
        )
        east = Curve.from_tangents(
            start=Vec(w, 0, 3.0), tan_start=Vec(0, 1, 0),
            end=Vec(w, d, 3.5), tan_end=Vec(0, 1, 0), scale=3.0,
        )
        north = Curve.from_tangents(
            start=Vec(w, d, 3.5), tan_start=Vec(-1, 0, 0),
            end=Vec(0, d, 3.2), tan_end=Vec(-1, 0, 0), scale=3.0,
        )
        west = Curve.from_tangents(
            start=Vec(0, d, 3.2), tan_start=Vec(0, -1, 0),
            end=Vec(0, 0, 2.0), tan_end=Vec(0, -1, 0), scale=3.0,
        )
        return [south, east, north, west]

    @pytest.mark.skipif(not _occ_available(), reason="needs pythonocc for the ribbon normals")
    def test_g1_rows_satisfied_against_occ_ribbons(self):
        from geometrykit.geometry.surface import Surface

        curves = self._network()
        params_u, params_v, oriented = patch_np.coons_parametrization(curves, nu=12, nv=12)

        ribbons = {
            "south": Surface.ribbon(curves[0], angle_deg=0, width=1.0),
            "east": Surface.ribbon(curves[1], angle_deg=0, width=1.0),
            "north": Surface.ribbon(curves[2], angle_deg=180, width=1.0),
            "west": Surface.ribbon(curves[3], angle_deg=0, width=1.0),
        }

        def ribbon_normal(edge, curve_index):
            curve = curves[curve_index]
            patch_params = params_u if edge in ("south", "north") else params_v
            curve_params = patch_np.edge_curve_params(oriented, edge, patch_params)

            def at(t):
                index = min(range(len(curve_params)), key=lambda k: abs(curve_params[k] - t))
                t_curve = curve_params[index]
                base = curve.point_at(t_curve)
                best = None
                for v_edge in (0.0, 1.0):
                    p = ribbons[edge].point_at(t_curve, v_edge)
                    dist = abs(p - base)
                    if best is None or dist < best[0]:
                        best = (dist, v_edge)
                dist, v_edge = best
                assert dist < 1e-4, f"{edge}: ribbon misses the curve at t={t_curve}"
                n = ribbons[edge].normal_at(t_curve, v_edge)
                ln = math.sqrt(n.x**2 + n.y**2 + n.z**2)
                assert ln > 1e-9
                return (n.x / ln, n.y / ln, n.z / ln)

            return at

        fields = {
            "south": ribbon_normal("south", 0),
            "east": ribbon_normal("east", 1),
            "north": ribbon_normal("north", 2),
            "west": ribbon_normal("west", 3),
        }
        surface, _corners, stations = patch_g1.constrained_coons_patch(
            curves, fields, nu=12, nv=12
        )
        for edge in patch_g1.EDGE_NAMES:
            params = stations[0] if edge in ("south", "north") else stations[1]
            residual = _g1_residual(surface, edge, params, fields[edge])
            assert residual < 1e-6, (edge, residual)

    @pytest.mark.skipif(not _occ_available(), reason="needs pythonocc for the twin")
    def test_kernel_patch_stays_near_the_occ_twin(self):
        """Shape check against the OCCT MakeFilling twin on the same network.

        Different construction (Coons + tangent-plane rows vs. OCC's
        minimising filling), so this bounds the *deviation*, not equality.
        """
        from geometrykit.geometry.surface import Surface

        curves = self._network()
        params_u, params_v, oriented = patch_np.coons_parametrization(curves, nu=12, nv=12)
        ribbons = {
            "south": Surface.ribbon(curves[0], angle_deg=0, width=1.0),
            "east": Surface.ribbon(curves[1], angle_deg=0, width=1.0),
            "north": Surface.ribbon(curves[2], angle_deg=180, width=1.0),
            "west": Surface.ribbon(curves[3], angle_deg=0, width=1.0),
        }
        twin = Surface.patch(
            curves=list(curves),
            constraints=["G1", "G1", "G1", "G1"],
            supports=[ribbons["south"], ribbons["east"], ribbons["north"], ribbons["west"]],
            tol=1e-4,
        )

        def ribbon_normal(edge, curve_index):
            curve = curves[curve_index]
            patch_params = params_u if edge in ("south", "north") else params_v
            curve_params = patch_np.edge_curve_params(oriented, edge, patch_params)

            def at(t):
                index = min(range(len(curve_params)), key=lambda k: abs(curve_params[k] - t))
                t_curve = curve_params[index]
                base = curve.point_at(t_curve)
                best = None
                for v_edge in (0.0, 1.0):
                    p = ribbons[edge].point_at(t_curve, v_edge)
                    dist = abs(p - base)
                    if best is None or dist < best[0]:
                        best = (dist, v_edge)
                dist, v_edge = best
                assert dist < 1e-4
                n = ribbons[edge].normal_at(t_curve, v_edge)
                ln = math.sqrt(n.x**2 + n.y**2 + n.z**2)
                return (n.x / ln, n.y / ln, n.z / ln)

            return at

        fields = {
            "south": ribbon_normal("south", 0),
            "east": ribbon_normal("east", 1),
            "north": ribbon_normal("north", 2),
            "west": ribbon_normal("west", 3),
        }
        surface, _corners, _stations = patch_g1.constrained_coons_patch(
            curves, fields, nu=12, nv=12
        )

        # both surfaces interpolate the same four curves: compare the
        # interior deviation only (boundaries agree by construction).
        worst = 0.0
        for i in range(1, 12):
            for j in range(1, 12):
                u, v = i / 12.0, j / 12.0
                p_ours = surface.point_at(u, v)
                p_twin, (uu, vv) = twin.project(p_ours)
                worst = max(worst, abs(p_ours - p_twin))
        assert worst < 0.25, worst