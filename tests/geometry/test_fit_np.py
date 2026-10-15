# This file was generated with the assistance of an AI coding tool.
"""M8: tolerance-driven adaptive approximation over a solved fill field.

Two tiers, matching the two environments:

- pure-python bookkeeping (knot expansion, segmentation, cut placement):
  runs everywhere, no numpy needed;
- the elimination-backed least-squares fit itself: needs numpy, so it is
  gated on import -- in practice this means the oracle env (which carries
  numpy alongside pythonocc), exactly where the field it approximates is
  built anyway.

The contract under test is the one the plan demands: the fitted surface
holds the field's *station* boundary lines (the M0 invariant survives the
representation change), the interior joins stay C1 (no segment knot may
rise above degree-1 multiplicity), and the sample residual respects the
tolerance that drove the cutting.
"""

from __future__ import annotations

import pytest
from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec
from geometrykit.runtimes._npu import fit_np, patch_np
from geometrykit.runtimes._npu.fit_np import (
    NoFitSpace,
    _cut_knots,
    _expand,
    _segments,
)

STEP = 1e-3


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


def _curvature_profile(surface, nu: int):
    us, vs = [], []
    for i in range(1, nu):
        for j in range(1, nu):
            u, v = i / nu, j / nu
            for which, store in (("v", vs), ("u", us)):
                if which == "v":
                    first = surface.tangents_at(u, v)[1]
                    later = surface.tangents_at(u, min(1.0, v + STEP))[1]
                else:
                    first = surface.tangents_at(u, v)[0]
                    later = surface.tangents_at(min(1.0, u + STEP), v)[0]
                normal = surface.normal_at(u, v)
                second = (
                    (later.x - first.x) / STEP,
                    (later.y - first.y) / STEP,
                    (later.z - first.z) / STEP,
                )
                speed = first.x**2 + first.y**2 + first.z**2
                curvature = (
                    second[0] * normal.x + second[1] * normal.y + second[2] * normal.z
                ) / speed
                store.append(abs(curvature))
    return (sum(us) / len(us), max(us), sum(vs) / len(vs), max(vs))


class TestKnotBookkeeping:
    """The segmentation machinery, without any numerics."""

    def test_expand_and_segments(self):
        flat = _expand([0.0, 0.5, 1.0], [5, 3, 5])
        assert len(flat) == 13
        assert _segments(flat) == [(0.0, 0.5), (0.5, 1.0)]
        assert _segments(_expand([0.0, 1.0], [5, 5])) == [(0.0, 1.0)]

    def test_cut_places_midpoints_under_the_worst_only(self):
        knots, mults = _cut_knots([0.0, 1.0], [5, 5], 4, [0.5], 1e-3)
        assert knots == [0.0, 0.5, 1.0]
        assert mults == [5, 3, 5]

    def test_cut_rests_below_tolerance(self):
        assert _cut_knots([0.0, 1.0], [5, 5], 4, [1e-4], 1e-3) is None

    def test_recut_never_rises_above_c1(self):
        """A cut node already at C1 multiplicity must not climb toward C0."""
        knots, mults = _cut_knots([0.0, 0.5, 1.0], [5, 3, 5], 4, [0.5, 0.5], 1e-3)
        assert 0.25 in knots and 0.75 in knots
        assert max(mults[1:-1]) == 3, mults

    def test_degree_zero_has_no_space(self):
        with pytest.raises(NoFitSpace):
            fit_np.adaptive_fit(None, degree=0)


try:
    import numpy  # noqa: F401

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

needs_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="the fitter needs numpy (oracle env)")


@needs_numpy
class TestStationExactFit:
    """The elimination-backed fit, on the plate field at probe density."""

    @staticmethod
    def _field(nu=12):
        curves = _network()
        field, _c = patch_np.discrete_plate_fill(
            curves, nu=nu, nv=nu, parametrization="arclength"
        )
        params_u, params_v, _o = patch_np.coons_parametrization(
            curves, nu=nu, nv=nu, parametrization="arclength"
        )
        return curves, field, params_u, params_v

    def test_boundary_stations_survive_the_fit(self):
        curves, field, params_u, params_v = self._field()
        surface, report = fit_np.adaptive_fit(
            field, degree=4, tol=1e-3, stations=(params_u, params_v)
        )
        assert report["boundary_worst"] < 1e-10, report
        for edge, spine in zip(("south", "east", "north", "west"), curves):
            params = params_u if edge in ("south", "north") else params_v
            curve_params = patch_np.edge_curve_params(
                patch_np.coons_parametrization(
                    curves, nu=12, nv=12, parametrization="arclength"
                )[2],
                edge,
                params,
            )
            for k, t in enumerate(params):
                if edge == "south":
                    got = surface.point_at(t, 0.0)
                elif edge == "north":
                    got = surface.point_at(t, 1.0)
                elif edge == "west":
                    got = surface.point_at(0.0, t)
                else:
                    got = surface.point_at(1.0, t)
                assert abs(got - spine.point_at(curve_params[k])) < 1e-9, (edge, k)

    def test_sample_residual_respects_the_tolerance(self):
        _curves, field, params_u, params_v = self._field()
        _surface, report = fit_np.adaptive_fit(
            field, degree=4, tol=1e-3, stations=(params_u, params_v)
        )
        assert report["sample_worst"] <= 1e-3, report

    def test_interior_joins_stay_c1(self):
        _curves, field, params_u, params_v = self._field()
        surface, _report = fit_np.adaptive_fit(
            field, degree=4, tol=1e-3, stations=(params_u, params_v)
        )
        assert max(surface.umults[1:-1]) <= surface.udegree - 1
        assert max(surface.vmults[1:-1]) <= surface.vdegree - 1

    def test_fit_is_smoother_than_a_bilinear_field(self):
        """The approximated surface must not ripple: its curvature profile
        stays in the same band as the field it approximates."""
        _curves, field, params_u, params_v = self._field()
        surface, _report = fit_np.adaptive_fit(
            field, degree=4, tol=1e-3, stations=(params_u, params_v)
        )
        before = _curvature_profile(field, 12)
        after = _curvature_profile(surface, 12)
        for name, a, b in zip(
            ("k_u mean", "k_u max", "k_v mean", "k_v max"), before, after
        ):
            assert b < 3.0 * a, (name, a, b)
