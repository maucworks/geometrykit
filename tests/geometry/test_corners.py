# This file was generated with the assistance of an AI coding tool.
"""M5: corners reconciled through the cage datum, mismatch reported.

Two tiers:

- pure protocol (analytic fields, no supports): the report structure, the
  Corner datum with both terminals, conflict detection when one edge's
  plane tilts away -- runs everywhere;
- the flowing roof (oracle ribbons): compatible corners stay unflagged,
  and the adjacent-station residual sits within 10x of the interior --
  the plan's gate, against the 2.8e-02 the band showed before the rows
  were extended over it.
"""

from __future__ import annotations

import math

import pytest
from geometrykit.cage import Corner
from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec
from geometrykit.runtimes._npu import patch_g1, patch_np


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


def _flat_fields(tilt_south_deg: float = 0.0) -> dict:
    tilt = math.radians(tilt_south_deg)
    normal_south = (0.0, -math.sin(tilt), math.cos(tilt))
    return {
        "south": {
            "normal": lambda t, n=normal_south: n,
            "slope": lambda t: (1.0, 0.0, 0.0),
        },
        "east": {
            "normal": lambda t: (0.0, 0.0, 1.0),
            "slope": lambda t: (0.0, 1.0, 0.0),
        },
        "north": {
            "normal": lambda t: (0.0, 0.0, 1.0),
            "slope": lambda t: (1.0, 0.0, 0.0),
        },
        "west": {
            "normal": lambda t: (0.0, 0.0, 1.0),
            "slope": lambda t: (0.0, 1.0, 0.0),
        },
    }


def _setup():
    curves = _network()
    params_u, params_v, oriented = patch_np.coons_parametrization(
        curves, nu=12, nv=12, parametrization="arclength"
    )
    return curves, params_u, params_v, oriented


class TestCornerReport:
    """The protocol itself, on analytic fields."""

    def test_compatible_corners_stay_unflagged(self):
        curves, params_u, params_v, oriented = _setup()
        report = patch_g1.corner_report(
            curves, oriented, params_u, params_v, _flat_fields()
        )
        assert set(report["corners"]) == {"SW", "SE", "NE", "NW"}
        assert report["conflict"] is False
        for entry in report["corners"].values():
            assert entry["plane_angle_deg"] < 1e-9
            assert entry["conflict"] is False

    def test_a_tilted_edge_flags_only_its_corners(self):
        curves, params_u, params_v, oriented = _setup()
        report = patch_g1.corner_report(
            curves, oriented, params_u, params_v, _flat_fields(tilt_south_deg=2.0)
        )
        assert report["conflict"] is True
        assert report["corners"]["SW"]["conflict"] is True
        assert report["corners"]["SE"]["conflict"] is True
        assert report["corners"]["NW"]["conflict"] is False
        assert report["corners"]["NE"]["conflict"] is False
        assert abs(report["corners"]["SW"]["plane_angle_deg"] - 2.0) < 1e-9

    def test_datum_is_a_cage_corner_with_both_terminals(self):
        curves, params_u, params_v, oriented = _setup()
        report = patch_g1.corner_report(
            curves, oriented, params_u, params_v, _flat_fields()
        )
        for name, entry in report["corners"].items():
            datum = entry["datum"]
            assert isinstance(datum, Corner), name
            assert set(datum.angles) == set(entry["edges"]), name
            # the frame sits on the corner and both terminals recover
            # unit tangents through it
            for edge in entry["edges"]:
                recovered = datum.tangent_for(edge)
                assert abs(sum(c * c for c in recovered) - 1.0) < 1e-9, (name, edge)

    def test_measured_residuals_present_with_a_surface(self):
        curves, params_u, params_v, oriented = _setup()
        surface, _c = patch_np.discrete_coons_patch(
            curves, nu=12, nv=12, parametrization="arclength"
        )
        report = patch_g1.corner_report(
            curves, oriented, params_u, params_v, _flat_fields(), surface=surface
        )
        for entry in report["corners"].values():
            for key, value in entry["measured"].items():
                assert value >= 0.0, (key, value)

    def test_missing_edge_is_skipped_not_failed(self):
        curves, params_u, params_v, oriented = _setup()
        fields = _flat_fields()
        del fields["north"]
        report = patch_g1.corner_report(
            curves, oriented, params_u, params_v, fields
        )
        assert set(report["corners"]) == {"SW", "SE"}


def _occ_available() -> bool:
    try:
        import OCC.Core.BRep  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _occ_available(), reason="needs pythonocc for ribbons")
class TestRoofCorners:
    """The plan's M5 gate on the flowing roof."""

    @staticmethod
    def _roof(nu=16):
        from geometrykit.geometry.surface import Surface

        curves = _network()
        supports = [
            Surface.ribbon(curves[0], angle_deg=0, width=1.0, symmetric=True),
            Surface.ribbon(curves[1], angle_deg=0, width=1.0, symmetric=True),
            Surface.ribbon(
                curves[2], angles_deg=[180, 155, 180], width=1.0, symmetric=True
            ),
            Surface.ribbon(curves[3], angle_deg=0, width=1.0, symmetric=True),
        ]
        params_u, params_v, oriented = patch_np.coons_parametrization(
            curves, nu=nu, nv=nu, parametrization="arclength"
        )
        raw = {
            edge: patch_g1.support_constraint(support, spine, towards=Vec(4, 3, 2.9))
            for edge, spine, support in zip(patch_g1.EDGE_NAMES, curves, supports)
        }
        # patch-parameter semantics, the way the export driver wraps specs:
        # INDEX-based (station k sits over curve_params[k]); value-based
        # min() mirrors reversed edges onto the wrong end.
        fields = {}
        for edge in patch_g1.EDGE_NAMES:
            patch_params = params_u if edge in ("south", "north") else params_v
            curve_params = patch_np.edge_curve_params(oriented, edge, patch_params)
            spec = raw[edge]

            def _at(t, _pp=patch_params, _cp=curve_params):
                k = min(range(len(_pp)), key=lambda m: abs(_pp[m] - t))
                return _cp[k]

            fields[edge] = {
                "normal": (lambda t, _f=_at, _s=spec: _s["normal"](_f(t))),
                "slope": (lambda t, _f=_at, _s=spec: _s["slope"](_f(t))),
                "support": spec["support"],
                "spine": spec["spine"],
            }
        return curves, params_u, params_v, oriented, fields

    def test_no_corner_conflicts_on_the_roof(self):
        curves, params_u, params_v, oriented, fields = self._roof()
        report = patch_g1.corner_report(
            curves, oriented, params_u, params_v, fields
        )
        assert report["conflict"] is False
        for name, entry in report["corners"].items():
            assert entry["plane_angle_deg"] < 0.01, (name, entry)

    def test_adjacent_residual_within_10x_of_interior(self):
        """The M5 gate on the shipped configuration (NU=16, G2, tangency).

        The band the rows used to skip now matches the interior. (At NU=12
        the G2 wide band contorts the corner region regardless of the
        corner band -- pre-existing, density-dependent, recorded in the
        plan; the shipped density is what this gates.)
        """
        curves, params_u, params_v, oriented, fields = self._roof(nu=16)
        # driver opts in explicitly: verified at this density (NU=16),
        # fragile below it (see the corner_stations contract)
        surface, _c, _s = patch_g1.constrained_coons_patch(
            curves, fields, nu=16, nv=16, tangency=0.75, g2=True, fill=True,
            corner_stations=0, parametrization="arclength",
        )
        report = patch_g1.corner_report(
            curves, oriented, params_u, params_v, fields, surface=surface
        )
        interior = 0.0
        for edge in patch_g1.EDGE_NAMES:
            params = params_u if edge in ("south", "north") else params_v
            for t in params[2:-2]:
                if edge == "south":
                    d = surface.tangents_at(t, 0.0)[1]
                elif edge == "north":
                    d = surface.tangents_at(t, 1.0)[1]
                elif edge == "west":
                    d = surface.tangents_at(0.0, t)[0]
                else:
                    d = surface.tangents_at(1.0, t)[0]
                nrm = fields[edge]["normal"](t)
                interior = max(
                    interior, abs(d.x * nrm[0] + d.y * nrm[1] + d.z * nrm[2])
                )
        adjacent = 0.0
        for entry in report["corners"].values():
            for key, value in entry["measured"].items():
                if "adjacent" in key:
                    adjacent = max(adjacent, value)
        assert adjacent <= 10.0 * max(interior, 1e-12), (adjacent, interior)
