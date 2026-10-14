# This file was generated with the assistance of an AI coding tool.
"""M2: the energy-minimising plate fill, and the gate it has to pass.

The gate is the point of this file: the fill must be *smoother than the Coons
base in every measured sense*. That is the check the earlier attempt failed by
40x, and a fill that only looks different is not a fill.

Two analytic anchors come first, because they decide whether the operator and
the boundary closure are right at all:

- a bilinear datum ``x*y`` satisfies both the plate equation and the simply
  supported closure, so the fill must reproduce it exactly;
- the boundary must come back exactly through the station map (the M0
  invariant), since the fill only changes the interior.
"""

from __future__ import annotations

import pytest
from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec
from geometrykit.runtimes._npu import linalg_np, patch_np

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
    """(|k_u| mean, |k_u| max, |k_v| mean, |k_v| max) over the interior."""
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
    return (
        sum(us) / len(us), max(us), sum(vs) / len(vs), max(vs)
    )


class TestPlateSolve:
    """The solver and its boundary closure, analytically."""

    def test_reproduces_a_bilinear_datum(self):
        """x*y satisfies the plate equation and the closure, so it is exact."""
        nx = ny = 8
        values = [
            (float(i * j), 0.0, 0.0) for i in range(nx + 1) for j in range(ny + 1)
        ]
        filled = linalg_np.plate_fill(values, nx, ny)
        worst = max(
            abs(filled[i * (ny + 1) + j][0] - float(i * j))
            for i in range(nx + 1)
            for j in range(ny + 1)
        )
        assert worst < 1e-9, worst

    def test_leaves_the_boundary_ring_untouched(self):
        nx = ny = 6
        values = [
            (float(i), float(j), 0.0) for i in range(nx + 1) for j in range(ny + 1)
        ]
        filled = linalg_np.plate_fill(values, nx, ny)
        for i in range(nx + 1):
            for j in range(ny + 1):
                if i in (0, nx) or j in (0, ny):
                    assert filled[i * (ny + 1) + j] == values[i * (ny + 1) + j]

    def test_smooths_a_bumpy_interior(self):
        """A deliberately spiky interior must come out smoother than it went in."""
        nx = ny = 10
        values = []
        for i in range(nx + 1):
            for j in range(ny + 1):
                spike = 1.0 if (i, j) == (5, 5) else 0.0
                values.append((float(i), float(j), spike))
        filled = linalg_np.plate_fill(values, nx, ny)

        def roughness(grid):
            total = 0.0
            for i in range(1, nx):
                for j in range(1, ny):
                    centre = grid[i * (ny + 1) + j][2]
                    total += abs(
                        grid[(i - 1) * (ny + 1) + j][2]
                        - 2.0 * centre
                        + grid[(i + 1) * (ny + 1) + j][2]
                    )
                    total += abs(
                        grid[i * (ny + 1) + j - 1][2]
                        - 2.0 * centre
                        + grid[i * (ny + 1) + j + 1][2]
                    )
            return total

        assert roughness(filled) < roughness(values) / 10.0


class TestPlateFillPatch:
    """The fill at patch level: boundary exact, interior smoother than Coons."""

    @pytest.mark.parametrize("mode", ["chord", "arclength"])
    def test_boundary_is_reproduced_exactly(self, mode):
        curves = _network()
        surface, _corners = patch_np.discrete_plate_fill(
            curves, nu=8, nv=8, parametrization=mode
        )
        params_u, params_v, oriented = patch_np.coons_parametrization(
            curves, nu=8, nv=8, parametrization=mode
        )
        for edge, spine in zip(("south", "east", "north", "west"), curves):
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
                assert abs(got - spine.point_at(curve_params[k])) < 1e-12, (edge, k)

    @pytest.mark.parametrize("mode", ["chord", "arclength"])
    def test_fill_is_smoother_than_the_coons_base(self, mode):
        """The M2 gate: smoother in every measure, both directions."""
        curves = _network()
        coons, _c = patch_np.discrete_coons_patch(
            curves, nu=12, nv=12, parametrization=mode
        )
        plate, _c = patch_np.discrete_plate_fill(
            curves, nu=12, nv=12, parametrization=mode
        )
        coons_profile = _curvature_profile(coons, 12)
        plate_profile = _curvature_profile(plate, 12)
        for name, before, after in zip(
            ("k_u mean", "k_u max", "k_v mean", "k_v max"), coons_profile, plate_profile
        ):
            assert after <= before, (mode, name, before, after)
        # and the cross-direction mean, which is where the Coons blend was worst
        assert plate_profile[2] < 0.6 * coons_profile[2], (mode, coons_profile, plate_profile)

    def test_the_fill_routes_through_the_patch_entry_point(self):
        """`constrained_coons_patch(fill=True)` bases the patch on the plate.

        With no continuity fields the constraint solve has nothing to impose,
        so the result is the base itself -- which is what makes this a clean
        check that the fill is wired in and needs no OCCT support surfaces.
        """
        from geometrykit.runtimes._npu import patch_g1

        curves = _network()
        filled, _c, _s = patch_g1.constrained_coons_patch(
            curves, {}, nu=8, nv=8, fill=True, parametrization="arclength"
        )
        plate, _c = patch_np.discrete_plate_fill(
            curves, nu=8, nv=8, parametrization="arclength"
        )
        worst = max(
            abs(filled.point_at(i / 8, j / 8) - plate.point_at(i / 8, j / 8))
            for i in range(9)
            for j in range(9)
        )
        assert worst < 1e-12, worst
