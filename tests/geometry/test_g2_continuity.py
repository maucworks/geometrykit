# This file was generated with the assistance of an AI coding tool.
"""G2 (curvature) continuity: the second-order constraint across the edges.

G1 matches the tangent plane; G2 matches the normal-section curvature across
the boundary as well, so the surface does not bend away from the support's
plane the moment it leaves the edge.

The load-bearing test is the second-derivative weight formula: everything else
is bookkeeping on top of it. Then the property -- the curvature gap closes,
and G1 survives the second pass.
"""

from __future__ import annotations

import pytest
from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec
from geometrykit.geometry.surface import Surface
from geometrykit.runtimes._npu import patch_g1, patch_np


def _occ_available() -> bool:
    try:
        import OCC.Core.BRep  # noqa: F401
        return True
    except ImportError:
        return False


class TestSecondDerivativeWeights:
    """The formula the curvature rows are built from."""

    def test_weights_match_a_numerical_second_derivative(self):
        """S''(0) from the weights equals a second difference of S itself.

        (`Curve.tangent_at` returns a *unit* tangent, so differencing it would
        not give a second derivative -- the position is the reference.)
        """
        curve = Curve.from_points(
            [Vec(0, 0, 0), Vec(1, 2, 0.5), Vec(2.5, 1.5, 1.0), Vec(4, 3, 0.2), Vec(5, 3.5, 0)],
            degree=3,
        )
        flat = []
        for knot, mult in zip(curve.knots, curve.multiplicities):
            flat.extend([float(knot)] * int(mult))
        low, high = patch_g1._second_deriv_weights(flat, curve.degree)

        for end, weights in (("low", low), ("high", high)):
            weighted = [0.0, 0.0, 0.0]
            for index, weight in weights.items():
                point = curve.points[index]
                weighted[0] += weight * point.x
                weighted[1] += weight * point.y
                weighted[2] += weight * point.z

            step = 1e-5
            if end == "low":
                first = curve.point_at(0.0)
                middle = curve.point_at(step)
                last = curve.point_at(2 * step)
            else:
                first = curve.point_at(1.0)
                middle = curve.point_at(1.0 - step)
                last = curve.point_at(1.0 - 2 * step)
            numeric = [
                (first.x - 2 * middle.x + last.x) / (step * step),
                (first.y - 2 * middle.y + last.y) / (step * step),
                (first.z - 2 * middle.z + last.z) / (step * step),
            ]
            for computed, expected in zip(weighted, numeric):
                assert computed == pytest.approx(expected, abs=1e-2), (end, weighted, numeric)

    def test_weights_sum_to_zero(self):
        """A second derivative annihilates constants, so the weights sum to 0."""
        curve = Curve.from_points(
            [Vec(i, 0, 0) for i in range(5)], degree=3
        )
        flat = []
        for knot, mult in zip(curve.knots, curve.multiplicities):
            flat.extend([float(knot)] * int(mult))
        low, high = patch_g1._second_deriv_weights(flat, curve.degree)
        assert sum(low.values()) == pytest.approx(0.0, abs=1e-9)
        assert sum(high.values()) == pytest.approx(0.0, abs=1e-9)

    def test_bezier_second_difference(self):
        """On a single Bezier span the weights are the plain second difference."""
        low, _high = patch_g1._second_deriv_weights(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], 3
        )
        assert low[0] == pytest.approx(6.0)
        assert low[1] == pytest.approx(-12.0)
        assert low[2] == pytest.approx(6.0)


@pytest.mark.skipif(not _occ_available(), reason="needs pythonocc for the supports")
class TestG2AcrossTheEdges:
    @staticmethod
    def _setup():
        network = [
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
        supports = [
            Surface.ribbon(network[0], angle_deg=0, width=1.0, symmetric=True),
            Surface.ribbon(network[1], angle_deg=0, width=1.0, symmetric=True),
            Surface.ribbon(network[2], angles_deg=[180, 155, 180], width=1.0, symmetric=True),
            Surface.ribbon(network[3], angle_deg=0, width=1.0, symmetric=True),
        ]
        towards = Vec(
            sum(c.point_at(0.5).x for c in network) / 4,
            sum(c.point_at(0.5).y for c in network) / 4,
            sum(c.point_at(0.5).z for c in network) / 4,
        )
        params_u, params_v, oriented = patch_np.coons_parametrization(network, nu=12, nv=12)
        fields, curve_params = {}, {}
        for edge, spine, support in zip(patch_g1.EDGE_NAMES, network, supports):
            patch_params = params_u if edge in ("south", "north") else params_v
            cp = patch_np.edge_curve_params(oriented, edge, patch_params)
            curve_params[edge] = cp
            spec = patch_g1.support_constraint(support, spine, towards=towards)
            fields[edge] = {
                "normal": lambda t, cp=cp, f=spec["normal"]: f(
                    min(cp, key=lambda c: abs(c - t))
                ),
                "slope": lambda t, cp=cp, f=spec["slope"]: f(
                    min(cp, key=lambda c: abs(c - t))
                ),
                "support": support,
                "spine": spine,
            }
        return network, supports, fields, curve_params

    @staticmethod
    def _curvature_gap(surface, fields, stations, curve_params):
        """(worst curvature gap, worst G1 residual) over the enforced band."""
        step = 1e-5
        worst_gap = 0.0
        worst_g1 = 0.0
        for edge in patch_g1.EDGE_NAMES:
            params = stations[0] if edge in ("south", "north") else stations[1]
            curvature = patch_g1.support_curvature_field(
                fields[edge]["support"], fields[edge]["spine"]
            )
            for t in params[2:-2]:
                if edge in ("south", "north"):
                    v0 = 0.0 if edge == "south" else 1.0
                    delta = step if edge == "south" else -step
                    first = surface.tangents_at(t, v0)[1]
                    later = surface.tangents_at(t, v0 + delta)[1]
                    normal = surface.normal_at(t, v0)
                else:
                    u0 = 0.0 if edge == "west" else 1.0
                    delta = step if edge == "west" else -step
                    first = surface.tangents_at(u0, t)[0]
                    later = surface.tangents_at(u0 + delta, t)[0]
                    normal = surface.normal_at(u0, t)
                second = (
                    (later.x - first.x) / delta,
                    (later.y - first.y) / delta,
                    (later.z - first.z) / delta,
                )
                speed = first.x**2 + first.y**2 + first.z**2
                patch_kappa = (
                    second[0] * normal.x + second[1] * normal.y + second[2] * normal.z
                ) / speed
                support_kappa = curvature(
                    min(curve_params[edge], key=lambda c: abs(c - t))
                )
                worst_gap = max(worst_gap, abs(patch_kappa - support_kappa))
                n = fields[edge]["normal"](t)
                worst_g1 = max(
                    worst_g1,
                    abs(first.x * n[0] + first.y * n[1] + first.z * n[2]),
                )
        return worst_gap, worst_g1

    def test_g2_closes_the_curvature_gap(self):
        network, _supports, fields, curve_params = self._setup()
        g1_only, _c, stations = patch_g1.constrained_coons_patch(
            network, fields, nu=12, nv=12, tangency=1.0
        )
        gap_before, _g1 = self._curvature_gap(g1_only, fields, stations, curve_params)
        with_g2, _c, stations = patch_g1.constrained_coons_patch(
            network, fields, nu=12, nv=12, tangency=1.0, g2=True
        )
        gap_after, worst_g1 = self._curvature_gap(with_g2, fields, stations, curve_params)
        assert gap_before > 0.01, gap_before
        assert gap_after < 1e-3, gap_after
        assert gap_after < gap_before / 100.0
        assert worst_g1 < 1e-9, worst_g1

    def test_g2_keeps_the_tangent_plane(self):
        network, _supports, fields, curve_params = self._setup()
        for tangency in (None, 1.0):
            surface, _c, stations = patch_g1.constrained_coons_patch(
                network, fields, nu=12, nv=12, tangency=tangency, g2=True
            )
            _gap, worst_g1 = self._curvature_gap(surface, fields, stations, curve_params)
            assert worst_g1 < 1e-9, (tangency, worst_g1)

    def test_g2_needs_the_support_in_the_spec(self):
        network, _supports, fields, _curve_params = self._setup()
        bare = {edge: fields[edge]["normal"] for edge in patch_g1.EDGE_NAMES}
        with pytest.raises(patch_g1.G1Error, match="G2 constraint"):
            patch_g1.constrained_coons_patch(network, bare, nu=12, nv=12, g2=True)

    def test_g2_does_not_fight_the_tangency_strength(self):
        """Both knobs on at once: curvature matched at any strength."""
        network, _supports, fields, curve_params = self._setup()
        for tangency in (0.5, 2.0):
            surface, _c, stations = patch_g1.constrained_coons_patch(
                network, fields, nu=12, nv=12, tangency=tangency, g2=True
            )
            gap, worst_g1 = self._curvature_gap(surface, fields, stations, curve_params)
            assert gap < 1e-3, (tangency, gap)
            assert worst_g1 < 1e-9, (tangency, worst_g1)
