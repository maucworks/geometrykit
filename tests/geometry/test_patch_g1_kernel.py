# This file was generated with the assistance of an AI coding tool.
"""P-1 G1 patch tests: coplanar demo (plain env) + flowing-roof fixture (oracle env)."""

from __future__ import annotations

import math

import pytest

from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec
from geometrykit.runtimes._npu import patch_np, patch_g1
from geometrykit.runtimes._npu.curve_np import chord_params


def _flowing_curves_mixed(heights=(1.5, 2.5, 4.0, 2.5, 1.5)):
    """The flowing-roof network with **mixed heights** (the non-planar fixture)."""
    h_sw, h_se, h_nw, h_ne = heights[0], heights[1], heights[2], heights[3]
    w = 8.0
    c_south = Curve.from_tangents(
        start=Vec(0, 0, h_sw), tan_start=Vec(1, 0, 0),
        end=Vec(w, 0, h_se), tan_end=Vec(1, 0, 0), scale=3.0,
    )
    c_east = Curve.from_tangents(
        start=Vec(w, 0, h_se), tan_start=Vec(0, 1, 0),
        end=Vec(w, 6, h_ne), tan_end=Vec(0, 1, 0), scale=3.0,
    )
    c_north = Curve.from_tangents(
        start=Vec(0, 6, h_nw), tan_start=Vec(1, 0, 0),
        end=Vec(w, 6, h_ne), tan_end=Vec(1, 0, 0), scale=3.0,
    )
    c_west = Curve.from_tangents(
        start=Vec(0, 6, h_nw), tan_start=Vec(0, -1, 0),
        end=Vec(0, 0, h_sw), tan_end=Vec(0, -1, 0), scale=3.0,
    )
    return [c_south, c_east, c_north, c_west]


def _occ_available():
    try:
        import OCC.Core.BRep  # noqa: F401
        return True
    except ImportError:
        return False


class TestCoplanarG1:
    """Plain env: the coplanar case — P-0 is already G1; the fix is a no-op."""

    def test_coplanar_g1_noop_preserves_planarity(self):
        curves = _flowing_curves_mixed(heights=(2.5, 2.5, 2.5, 2.5))
        normals = {
            "south": [(0, 0, 1.0)] * 11,
            "north": [(0, 0, 1.0)] * 11,
            "west": [(0, 0, 1.0)] * 11,
            "east": [(0, 0, 1.0)] * 11,
        }
        s, _ = patch_g1.constrained_coons_patch(curves, normals)
        zs = [
            s.point_at(i / 10.0, j / 10.0).z for i in range(11) for j in range(11)
        ]
        assert max(abs(z - 2.5) for z in zs) < 1e-9

    def test_row1_adjustment_changes_cross_derivative(self):
        """A tilted normal field turns the patch toward the plane (the fix works)."""
        curves = _flowing_curves_mixed(heights=(2.5, 2.5, 2.5, 2.5))
        tilt = 0.2
        normals = {
            "south": [(0.0, -tilt, math.sqrt(1 - tilt * tilt))] * 11,
            "north": [(0, 0, 1.0)] * 11,
            "west": [(0, 0, 1.0)] * 11,
            "east": [(0, 0, 1.0)] * 11,
        }
        s_p0, _ = patch_np.discrete_coons_patch(curves)
        s_p1, _ = patch_g1.constrained_coons_patch(curves, normals)
        eps = 1e-4
        for x_frac in (0.3, 0.5, 0.7):
            u = x_frac
            d_p0y = (s_p0.point_at(u, eps).y - s_p0.point_at(u, 0.0).y) / eps
            d_p1y = (s_p1.point_at(u, eps).y - s_p1.point_at(u, 0.0).y) / eps
            assert d_p1y < d_p0y, (x_frac, d_p0y, d_p1y)


class TestMixedHeightFlowing:
    """Oracle env: flowing-roof mixed-height with OCC-sampled ribbon normals."""

    @pytest.mark.skipif(not _occ_available(), reason="needs pythonocc")
    def test_g1_constraint_satisfied_at_stations(self):
        curves = _flowing_curves_mixed()
        rb_specs = [
            ("south", curves[0], {"angle_deg": 0}, "u"),
            ("east", curves[1], {"angle_deg": 0}, "v"),
            ("north", curves[2], {"angle_deg": 180}, "u"),
            ("west", curves[3], {"angle_deg": 0}, "v"),
        ]
        from geometrykit.geometry.surface import Surface as GSurface

        # the OCC ribbon's u-parametrization matches the base curve's chord
        # params (validated by the shared-edge probe below)
        fields = {}
        patch_params = [k / 15.0 for k in range(16)]
        edge_boundaries = {"south": 0.0, "east": 1.0, "north": 1.0, "west": 0.0}
        for name, curve, kwargs, _cross_dir in rb_specs:
            r = GSurface.ribbon(curve, width=1.0, **kwargs)
            stations_t, normals = [], []
            for k in range(11):
                t = (k + 1) / 12  # interior data stations
                q = curve.point_at(t)
                best = None
                for ve in (0.0, 1.0):
                    p = r.point_at(t, ve)
                    dist = abs(p - q)
                    if best is None or dist < best[0]:
                        best = (dist, ve)
                dist, ve_s = best
                assert dist < 1e-4, f"{name}: ribbon does not share the edge at t={t}"
                n = r.normal_at(t, ve_s)
                ln = math.sqrt(n.x**2 + n.y**2 + n.z**2)
                stations_t.append(t)
                normals.append((n.x / ln, n.y / ln, n.z / ln))
            fields[name] = {"stations": stations_t, "normals": normals}

        s_p1, _ = patch_g1.constrained_coons_patch(
            curves,
            {name: f["normals"] for name, f in fields.items()},
            nu=16, nv=16,
        )

        eps = 1e-4
        for name, curve, _kwargs, cross_dir in rb_specs:
            field = fields[name]
            boundary = edge_boundaries[name]
            for k, t in enumerate(field["stations"]):
                nx, ny, nz = field["normals"][k]
                if cross_dir == "u":
                    u = t
                    v0, v1 = boundary, boundary + eps
                    d = (
                        (s_p1.point_at(u, v1).x - s_p1.point_at(u, v0).x) / eps,
                        (s_p1.point_at(u, v1).y - s_p1.point_at(u, v0).y) / eps,
                        (s_p1.point_at(u, v1).z - s_p1.point_at(u, v0).z) / eps,
                    )
                else:
                    u0, u1 = boundary, boundary + eps
                    v = t
                    d = (
                        (s_p1.point_at(u1, v).x - s_p1.point_at(u0, v).x) / eps,
                        (s_p1.point_at(u1, v).y - s_p1.point_at(u0, v).y) / eps,
                        (s_p1.point_at(u1, v).z - s_p1.point_at(u0, v).z) / eps,
                    )
                dot = d[0] * nx + d[1] * ny + d[2] * nz
                assert abs(dot) < 1e-4, (name, k, t, dot)
