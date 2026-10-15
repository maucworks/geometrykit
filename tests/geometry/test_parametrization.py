# This file was generated with the assistance of an AI coding tool.
"""M0 verification: the parameterisation contract, and what it buys.

Asserts the invariant that makes the parameterisation choice safe to change --
`surface.point_at(params[i], edge)` is `curve.point_at(stations[i])` -- and
records the two things arc-length stations buy: even iso-lines in space, and a
much lower curvature peak in the existing G1+G2 pipeline.

The control-*point* spacing is deliberately not asserted: the Greville
abscissae of a clamped B-spline cannot be made uniform (uniform knots give
1:2:3:...:3:2:1), and denser poles at the ends are an advantage for imposing
boundary conditions, not a defect.
"""

from __future__ import annotations

import pytest
from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec
from geometrykit.runtimes._npu import patch_np


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


class TestStationMap:
    """The mapping from patch parameter to curve parameter."""

    def test_chord_station_map_is_the_curve_itself(self):
        """Chord mode: the patch parameter IS the curve's own parameter."""
        curves = _network()
        params_u, _params_v, oriented = patch_np.coons_parametrization(
            curves, nu=8, nv=8, parametrization="chord"
        )
        stations = patch_np.edge_curve_params(oriented, "south", params_u)
        assert stations == pytest.approx(params_u, abs=1e-12)

    def test_arclength_stations_are_evenly_spaced_in_space(self):
        """Arc-length mode: equal gaps along the curve, whatever its speed."""
        curves = _network()
        params_u, _params_v, oriented = patch_np.coons_parametrization(
            curves, nu=8, nv=8, parametrization="arclength"
        )
        assert params_u == pytest.approx([i / 8 for i in range(9)], abs=1e-12)
        stations = patch_np.edge_curve_params(oriented, "south", params_u)
        points = [curves[0].point_at(t) for t in stations]
        gaps = [abs(points[i + 1] - points[i]) for i in range(len(points) - 1)]
        assert max(gaps) / min(gaps) < 1.02, max(gaps) / min(gaps)

    def test_the_invariant_holds_in_both_modes(self):
        """surface at a patch station == curve at its station (the contract)."""
        curves = _network()
        for mode in ("chord", "arclength"):
            params_u, params_v, oriented = patch_np.coons_parametrization(
                curves, nu=8, nv=8, parametrization=mode
            )
            surface, _corners = patch_np.discrete_coons_patch(
                curves, nu=8, nv=8, parametrization=mode
            )
            for edge, spine in zip(
                ("south", "east", "north", "west"), curves
            ):
                params = params_u if edge in ("south", "north") else params_v
                curve_params = patch_np.edge_curve_params(oriented, edge, params)
                for k, t in enumerate(params):
                    if edge == "south":
                        got = surface.point_at(t, 0.0)
                    elif edge == "north":
                        got = surface.point_at(t, 1.0)
                    elif edge == "west":
                        got = surface.point_at(0.0, t)
                    else:
                        got = surface.point_at(1.0, t)
                    assert abs(got - spine.point_at(curve_params[k])) < 1e-12, (
                        mode,
                        edge,
                        k,
                    )

    def test_uniform_knots_only_in_arclength_mode(self):
        """Uniform parameters need uniform knots; averaged knots keep a 2x span."""
        curves = _network()
        for mode, expected_ratio in (("chord", 1.5), ("arclength", 1.0001)):
            surface, _c = patch_np.discrete_coons_patch(
                curves, nu=8, nv=8, parametrization=mode
            )
            flat = []
            for knot, mult in zip(surface.uknots, surface.umults):
                flat.extend([knot] * mult)
            distinct = sorted(set(flat))
            spans = [distinct[i + 1] - distinct[i] for i in range(len(distinct) - 1)]
            ratio = max(spans) / min(spans)
            if mode == "chord":
                assert ratio > expected_ratio, ratio
            else:
                assert ratio < expected_ratio, ratio

    def test_unknown_mode_raises(self):
        with pytest.raises(patch_np.PatchError, match="unknown parametrization"):
            patch_np.coons_parametrization(_network(), parametrization="uniform")
