# This file was generated with the assistance of an AI coding tool.
"""Tangency strength: the magnitude G1 leaves free, made an explicit knob.

G1 fixes the tangent *plane* and nothing else -- how stiffly the patch leaves
the edge is the solver's leftover. `tangency` names it: the patch's boundary
slope as a multiple of the support's own, measured per unit distance so it
does not depend on either surface's parameterisation.

The numbers asserted here are the property, not a golden value: strength 1.0
must reproduce the support's own slope, and the response must be linear.
"""

from __future__ import annotations

import math

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


def _network():
    """The flowing-roof front patch, mixed heights."""
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


@pytest.mark.skipif(not _occ_available(), reason="needs pythonocc for the supports")
class TestTangencyStrength:
    @staticmethod
    def _setup():
        network = _network()
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
        fields = {}
        for edge, spine, support in zip(patch_g1.EDGE_NAMES, network, supports):
            patch_params = params_u if edge in ("south", "north") else params_v
            curve_params = patch_np.edge_curve_params(oriented, edge, patch_params)
            spec = patch_g1.support_constraint(support, spine, towards=towards)
            fields[edge] = {
                "normal": lambda t, cp=curve_params, f=spec["normal"]: f(
                    min(cp, key=lambda c: abs(c - t))
                ),
                "slope": lambda t, cp=curve_params, f=spec["slope"]: f(
                    min(cp, key=lambda c: abs(c - t))
                ),
                "support": support,
                "spine": spine,
            }
        return network, supports, fields

    @staticmethod
    def _boundary_slopes(surface, fields, stations):
        """(patch slope per unit distance, worst |d.N|) over the enforced band."""
        slopes = []
        worst = 0.0
        for edge in patch_g1.EDGE_NAMES:
            params = stations[0] if edge in ("south", "north") else stations[1]
            for t in params[2:-2]:
                if edge == "south":
                    d = surface.tangents_at(t, 0.0)[1]
                elif edge == "north":
                    d = surface.tangents_at(t, 1.0)[1]
                elif edge == "west":
                    d = surface.tangents_at(0.0, t)[0]
                else:
                    d = surface.tangents_at(1.0, t)[0]
                reach = patch_g1._cross_extent(surface, edge, t)
                slopes.append(abs(d) / max(reach, 1e-9))
                n = fields[edge]["normal"](t)
                worst = max(worst, abs(d.x * n[0] + d.y * n[1] + d.z * n[2]))
        return sum(slopes) / len(slopes), worst

    def test_strength_one_matches_the_support_own_slope(self):
        network, supports, fields = self._setup()
        surface, _c, stations = patch_g1.constrained_patch(
            network, fields, nu=12, nv=12, tangency=1.0
        )
        patch_slope, _worst = self._boundary_slopes(surface, fields, stations)

        support_slopes = []
        towards = Vec(
            sum(c.point_at(0.5).x for c in network) / 4,
            sum(c.point_at(0.5).y for c in network) / 4,
            sum(c.point_at(0.5).z for c in network) / 4,
        )
        for spine, support in zip(network, supports):
            _n, slope = patch_g1.support_fields(support, spine, towards=towards)(0.5)
            length = math.sqrt(sum(c * c for c in slope))
            reach = patch_g1._support_reach(support, spine, 0.5)
            support_slopes.append(length / reach)
        expected = sum(support_slopes) / len(support_slopes)
        assert patch_slope == pytest.approx(expected, rel=0.05), (
            patch_slope,
            expected,
        )

    def test_response_is_linear_in_the_strength(self):
        network, _supports, fields = self._setup()
        measured = {}
        for tangency in (0.5, 1.0, 2.0):
            surface, _c, stations = patch_g1.constrained_patch(
                network, fields, nu=12, nv=12, tangency=tangency
            )
            measured[tangency], _worst = self._boundary_slopes(surface, fields, stations)
        assert measured[1.0] == pytest.approx(2 * measured[0.5], rel=0.02)
        assert measured[2.0] == pytest.approx(2 * measured[1.0], rel=0.02)

    def test_the_default_is_softer_than_the_support(self):
        """Which is the whole point of the knob: min-norm under-delivers."""
        network, _supports, fields = self._setup()
        default, _c, stations = patch_g1.constrained_patch(
            network, fields, nu=12, nv=12
        )
        stiff, _c, stations = patch_g1.constrained_patch(
            network, fields, nu=12, nv=12, tangency=1.0
        )
        softer, _worst = self._boundary_slopes(default, fields, stations)
        harder, _worst = self._boundary_slopes(stiff, fields, stations)
        assert softer < harder

    def test_g1_holds_at_every_strength(self):
        network, _supports, fields = self._setup()
        for tangency in (0.25, 1.0, 4.0):
            surface, _c, stations = patch_g1.constrained_patch(
                network, fields, nu=12, nv=12, tangency=tangency
            )
            _slope, worst = self._boundary_slopes(surface, fields, stations)
            assert worst < 1e-6, (tangency, worst)

    def test_strength_without_a_slope_field_raises(self):
        network, _supports, fields = self._setup()
        bare = {edge: fields[edge]["normal"] for edge in patch_g1.EDGE_NAMES}
        with pytest.raises(patch_g1.G1Error, match="slope"):
            patch_g1.constrained_patch(
                network, bare, nu=12, nv=12, tangency=1.0
            )