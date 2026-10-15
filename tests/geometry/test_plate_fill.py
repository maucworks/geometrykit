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
from geometrykit.geometry.surface import Surface
from geometrykit.runtimes._npu import linalg_np, patch_g1, patch_np

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


class TestClampedPlateSolve:
    """The M3 closure: a side's cross-derivative as a condition of the solve.

    The ghost is eliminated by the slope relation, which is exact for affine
    slope data -- so ``x*y`` and ``x**2`` come back exactly under it, and so
    does a mixed grid where only some sides carry slope data. These are the
    checks that decide whether the boundary condition is *the* condition, not
    an approximation of it.
    """

    @staticmethod
    def _flat(f, nx, ny):
        return [(f(i, j), 0.0, 0.0) for i in range(nx + 1) for j in range(ny + 1)]

    @staticmethod
    def _slopes(dfx, dfy, nx, ny, step=1.0):
        """Per-station cross-derivatives in the increasing-index direction.

        The lambdas are in *real* coordinates, so the stations are scaled by
        the step before sampling.
        """
        return {
            "west": [(dfx(0.0, j * step), 0.0, 0.0) for j in range(ny + 1)],
            "east": [(dfx(nx * step, j * step), 0.0, 0.0) for j in range(ny + 1)],
            "south": [(dfy(i * step, 0.0), 0.0, 0.0) for i in range(nx + 1)],
            "north": [(dfy(i * step, ny * step), 0.0, 0.0) for i in range(nx + 1)],
        }

    @staticmethod
    def _assert_exact(name, f, dfx, dfy, nx=7, ny=7, step=1.0):
        values = [
            (f(i * step, j * step), 0.0, 0.0)
            for i in range(nx + 1)
            for j in range(ny + 1)
        ]
        slopes = TestClampedPlateSolve._slopes(dfx, dfy, nx, ny, step)
        filled = linalg_np.plate_fill(values, nx, ny, slopes=slopes, steps=(step, step))
        worst = max(
            abs(filled[i * (ny + 1) + j][0] - f(i * step, j * step))
            for i in range(nx + 1)
            for j in range(ny + 1)
        )
        assert worst < 1e-9, (name, worst)

    def test_reproduces_a_bilinear_datum_clamped(self):
        self._assert_exact(
            "x*y", lambda x, y: x * y,
            lambda x, y: y, lambda x, y: x,
        )

    def test_reproduces_a_quadratic_datum_clamped(self):
        self._assert_exact(
            "x^2", lambda x, y: x * x,
            lambda x, y: 2.0 * x, lambda x, y: 0.0,
        )

    def test_reproduces_a_sum_clamped(self):
        self._assert_exact(
            "x*y + 2 x^2", lambda x, y: x * y + 2.0 * x * x,
            lambda x, y: y + 4.0 * x, lambda x, y: x,
        )

    def test_exact_with_a_non_unit_step(self):
        step = 0.3804
        self._assert_exact(
            "x*y + 2 x^2 (step)", lambda x, y: x * y + 2.0 * x * x,
            lambda x, y: y + 4.0 * x, lambda x, y: x, step=step,
        )

    def test_mixed_closures(self):
        """Sides without slope data fall back to simply supported, and the
        clamped sides stay exact."""
        nx = ny = 7
        slopes = {
            "west": [(j, 0.0, 0.0) for j in range(ny + 1)],
            "east": [(j, 0.0, 0.0) for j in range(ny + 1)],
        }
        values = [
            (x * y + 0.0 * 0, x * 0.0, 0.0) if False else (x * y, 0.0, 0.0)
            for x in range(nx + 1) for y in range(ny + 1)
        ]
        filled = linalg_np.plate_fill(values, nx, ny, slopes=slopes)
        worst = max(
            abs(filled[i * (ny + 1) + j][0] - i * j)
            for i in range(nx + 1)
            for j in range(ny + 1)
        )
        assert worst < 1e-9, worst


class TestCurvaturePinnedSolve:
    """M4: a side with slope AND curvature data pins its first inner ring.

    Value, slope and curvature cost exactly one degree of freedom per
    boundary station at the plate stencil -- the ghost is consumed by the
    slope relation -- so the consistent scheme fixes ring 1 to the Hermite
    value ``u_edge + du*slope + du^2*kappa/2``. That is quadratic-exact:
    with data taken from a quadratic datum the pinned ring *is* the sampled
    ring and the fill reproduces the datum exactly. (The first attempt of
    this milestone had the pin without the curvature term and lost ``x**2``.)
    """

    @staticmethod
    def _data(f, dfx, dfy, dfxx, dfyy, nx=8, ny=6, step=1.0):
        values = [
            (f(i * step, j * step), 0.0, 0.0)
            for i in range(nx + 1)
            for j in range(ny + 1)
        ]
        slopes = {
            "west": [(dfx(0.0, j * step), 0.0, 0.0) for j in range(ny + 1)],
            "east": [(dfx(nx * step, j * step), 0.0, 0.0) for j in range(ny + 1)],
            "south": [(dfy(i * step, 0.0), 0.0, 0.0) for i in range(nx + 1)],
            "north": [(dfy(i * step, ny * step), 0.0, 0.0) for i in range(nx + 1)],
        }
        curvatures = {
            "west": [(dfxx(0.0, j * step), 0.0, 0.0) for j in range(ny + 1)],
            "east": [(dfxx(nx * step, j * step), 0.0, 0.0) for j in range(ny + 1)],
            "south": [(dfyy(i * step, 0.0), 0.0, 0.0) for i in range(nx + 1)],
            "north": [(dfyy(i * step, ny * step), 0.0, 0.0) for i in range(nx + 1)],
        }
        return values, slopes, curvatures

    @staticmethod
    def _exact(name, f, dfx, dfy, dfxx, dfyy, nx=8, ny=6, step=1.0,
               sides=("west", "east", "south", "north")):
        values, slopes, curvatures = TestCurvaturePinnedSolve._data(
            f, dfx, dfy, dfxx, dfyy, nx=nx, ny=ny, step=step
        )
        selected = {k: v for k, v in curvatures.items() if k in sides}
        filled = linalg_np.plate_fill(
            values, nx, ny,
            slopes=slopes,
            curvatures=selected,
            steps=(step, step),
        )
        worst = max(
            abs(filled[i * (ny + 1) + j][0] - f(i * step, j * step))
            for i in range(nx + 1)
            for j in range(ny + 1)
        )
        assert worst < 1e-9, (name, worst)

    def test_bilinear_with_pin(self):
        self._exact(
            "x*y", lambda x, y: x * y,
            lambda x, y: y, lambda x, y: x,
            lambda x, y: 0.0, lambda x, y: 0.0,
        )

    def test_quadratic_with_pin(self):
        self._exact(
            "x*y + 2x^2 + 3y^2", lambda x, y: x * y + 2.0 * x * x + 3.0 * y * y,
            lambda x, y: y + 4.0 * x, lambda x, y: x + 6.0 * y,
            lambda x, y: 4.0, lambda x, y: 6.0,
        )

    def test_mixed_pin_and_clamp(self):
        self._exact(
            "x*y + 2x^2 + 3y^2 (mixed)",
            lambda x, y: x * y + 2.0 * x * x + 3.0 * y * y,
            lambda x, y: y + 4.0 * x, lambda x, y: x + 6.0 * y,
            lambda x, y: 4.0, lambda x, y: 6.0,
            sides=("west", "east"),  # south/north keep the G1 clamp
        )

    def test_non_unit_step_with_pin(self):
        self._exact(
            "x*y + 2x^2 (step)",
            lambda x, y: x * y + 2.0 * x * x,
            lambda x, y: y + 4.0 * x, lambda x, y: x,
            lambda x, y: 4.0, lambda x, y: 0.0,
            step=0.3704,
        )

    def test_slope_only_datum_is_unchanged(self):
        """The M3 route must not have shifted: no curvature data, no pins."""
        self._exact(
            "x*y slope-only", lambda x, y: x * y,
            lambda x, y: y, lambda x, y: x,
            lambda x, y: 0.0, lambda x, y: 0.0,
            sides=(),  # curvatures dict empty -> clamped like M3
        )

    def test_pins_need_slope_on_the_side(self):
        values = [(float(i * j), 0.0, 0.0) for i in range(9) for j in range(7)]
        with pytest.raises(ValueError):
            linalg_np.plate_fill(
                values, 8, 6, curvatures={k: [(0.0, 0.0, 0.0)] * n for k, n in
                                           (("west", 7), ("east", 7), ("south", 9), ("north", 9))}
            )


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
        """`constrained_patch(fill=True)` bases the patch on the plate.

        With no continuity fields the constraint solve has nothing to impose,
        so the result is the base itself -- which is what makes this a clean
        check that the fill is wired in and needs no OCCT support surfaces.
        """
        from geometrykit.runtimes._npu import patch_g1

        curves = _network()
        filled, _c, _s = patch_g1.constrained_patch(
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


def _occ_available() -> bool:
    try:
        import OCC.Core.BRep  # noqa: F401
        return True
    except ImportError:
        return False


def _slope_ratio(surface, edge: str, t: float) -> float:
    """The boundary cross-tangent's magnitude per unit cross extent."""
    if edge == "south":
        cross = surface.tangents_at(t, 0.0)[1]
    elif edge == "north":
        cross = surface.tangents_at(t, 1.0)[1]
    elif edge == "west":
        cross = surface.tangents_at(0.0, t)[0]
    else:
        cross = surface.tangents_at(1.0, t)[0]
    extent = max(patch_g1._cross_extent(surface, edge, t), 1e-9)
    return abs(cross) / extent


@pytest.mark.skipif(not _occ_available(), reason="needs pythonocc for ribbons")
class TestCurvatureBeltOptIn:
    """M4's eliminated ring: exact machinery, but opt-in and measured worse.

    For flat ribbons the belts pin zero cross-curvature across the first two
    spans; the roof's own bend then lives one span inward, which roughens the
    interior and makes the G2 row pass diverge on the mixed-height network.
    The gates here assert only what the machinery guarantees -- boundary
    exactness, and that the opt-in path routes the belts -- not the shape.

    (The shape verdict lives in the research note and the plan record: G2
    stays with the row passes for near-flat supports.)
    """

    def _fields(self, support_table):
        curves = _network()
        params_u, params_v, oriented = patch_np.coons_parametrization(
            curves, nu=12, nv=12, parametrization="arclength"
        )
        curves = _network()
        fields = {}
        for edge, spine, support in zip(
            patch_g1.EDGE_NAMES, curves, support_table
        ):
            fields[edge] = patch_g1.support_constraint(
                support, spine, towards=Vec(4, 3, 2.9)
            )
        return curves, params_u, params_v, oriented, fields

    def test_belt_fill_keeps_the_boundary_exact(self):
        supports = [
            Surface.ribbon(c, angle_deg=0, width=1.0, symmetric=True)
            for c in _network()
        ]
        curves, params_u, params_v, oriented, fields = self._fields(supports)
        coons, _c = patch_np.discrete_coons_patch(
            curves, nu=12, nv=12, parametrization="arclength"
        )
        slopes = patch_g1.grid_edge_slopes(
            coons, oriented, params_u, params_v, fields, 1.0
        )
        curvatures = patch_g1.grid_edge_curvatures(
            coons, oriented, params_u, params_v, fields
        )
        fill, _corners = patch_np.discrete_plate_fill(
            curves, nu=12, nv=12, parametrization="arclength",
            slopes=slopes, curvatures=curvatures,
        )
        for edge, spine in zip(patch_g1.EDGE_NAMES, curves):
            params = params_u if edge in ("south", "north") else params_v
            curve_params = patch_np.edge_curve_params(oriented, edge, params)
            for k, t in enumerate(params):
                if edge == "south":
                    got = fill.point_at(t, 0.0)
                elif edge == "north":
                    got = fill.point_at(t, 1.0)
                elif edge == "west":
                    got = fill.point_at(0.0, t)
                else:
                    got = fill.point_at(1.0, t)
                assert abs(got - spine.point_at(curve_params[k])) < 1e-12, (edge, k)

    def test_entry_keeps_belts_opt_in(self):
        """g2=True with no g2_in_fill must not belt: the fill equals the
        slope-clamped one."""
        ridged = [
            Surface.ribbon(_network()[0], angle_deg=0, width=1.0, symmetric=True),
            Surface.ribbon(_network()[1], angle_deg=0, width=1.0, symmetric=True),
            Surface.ribbon(
                _network()[2], angles_deg=[180, 155, 180], width=1.0, symmetric=True
            ),
            Surface.ribbon(_network()[3], angle_deg=0, width=1.0, symmetric=True),
        ]
        curves, _pu, _pv, _o, fields = self._fields(ridged)
        surface, _c, _st = patch_g1.constrained_patch(
            curves, fields, nu=12, nv=12, tangency=1.0, g2=True, fill=True,
            parametrization="arclength",
        )
        reference, _c, _st = patch_g1.constrained_patch(
            curves, fields, nu=12, nv=12, tangency=1.0, g2=True, fill=True,
            g2_in_fill=True, parametrization="arclength",
        )
        same = max(
            abs(surface.point_at(i / 10, j / 10) - reference.point_at(i / 10, j / 10))
            for i in range(11)
            for j in range(11)
        )
        assert same > 1e-3, same  # the opt-in genuinely changes the patch
        exact = max(
            abs(surface.point_at(i / 10, 0.0) - reference.point_at(i / 10, 0.0))
            for i in range(11)
        )
        assert exact < 1e-9, exact  # ...but never the boundary


@pytest.mark.skipif(not _occ_available(), reason="needs pythonocc for ribbons")
class TestClampedFillCarriesTheSupportsSlope:
    """M3 on the flowing-roof network: G1 as a boundary condition of the fill.

    The gate this file exists to keep: the *fill itself* must arrive at the
    support's leaving rate, not a Coons-shape base that the row passes then
    bend into compliance. Pre-rows the rate should be near the support's own
    (2.000 per unit cross extent at tangency 1.0); post-rows it must be exact
    and the interior must stay smooth.
    """

    @staticmethod
    def _setup(nu: int = 12):
        supports = [
            Surface.ribbon(_network()[0], angle_deg=0, width=1.0, symmetric=True),
            Surface.ribbon(_network()[1], angle_deg=0, width=1.0, symmetric=True),
            Surface.ribbon(
                _network()[2], angles_deg=[180, 155, 180], width=1.0, symmetric=True
            ),
            Surface.ribbon(_network()[3], angle_deg=0, width=1.0, symmetric=True),
        ]
        network = _network()
        params_u, params_v, oriented = patch_np.coons_parametrization(
            network, nu=nu, nv=nu, parametrization="arclength"
        )
        fields = {
            edge: patch_g1.support_constraint(support, spine, towards=Vec(4, 3, 2.9))
            for edge, spine, support in zip(patch_g1.EDGE_NAMES, network, supports)
        }
        return network, params_u, params_v, oriented, fields

    def test_the_fill_carries_the_slope_before_any_correction(self):
        network, params_u, params_v, oriented, fields = self._setup()
        coons, _c = patch_np.discrete_coons_patch(
            network, nu=12, nv=12, parametrization="arclength"
        )
        slopes = patch_g1.grid_edge_slopes(
            coons, oriented, params_u, params_v, fields, 1.0
        )
        assert set(slopes) == {"south", "east", "north", "west"}
        fill, _corners = patch_np.discrete_plate_fill(
            network, nu=12, nv=12, parametrization="arclength", slopes=slopes
        )
        # boundary values untouched
        for edge, spine in zip(patch_g1.EDGE_NAMES, network):
            params = params_u if edge in ("south", "north") else params_v
            curve_params = patch_np.edge_curve_params(oriented, edge, params)
            for k, t in enumerate(params):
                if edge == "south":
                    got = fill.point_at(t, 0.0)
                elif edge == "north":
                    got = fill.point_at(t, 1.0)
                elif edge == "west":
                    got = fill.point_at(0.0, t)
                else:
                    got = fill.point_at(1.0, t)
                assert abs(got - spine.point_at(curve_params[k])) < 1e-12, (edge, k)
        # and the leaving rate comes from the solve, not from the rows
        for edge in patch_g1.EDGE_NAMES:
            params = params_u if edge in ("south", "north") else params_v
            ratios = [_slope_ratio(fill, edge, t) for t in params[1:-1]]
            assert sum(ratios) / len(ratios) > 1.7, (edge, ratios)
            assert min(ratios) > 1.5, (edge, ratios)

    def test_rows_confirm_and_the_interior_stays_smooth(self):
        network, _params_u, _params_v, _oriented, fields = self._setup()
        surface, _c, stations = patch_g1.constrained_patch(
            network, fields, nu=12, nv=12, tangency=1.0, g2=True, fill=True,
            parametrization="arclength",
        )
        # the rows finish the job: leaving rate exactly the support's own
        # (over the stations they are allowed to touch -- the corner band is
        # deliberately free, see corner_stations)
        for edge in patch_g1.EDGE_NAMES:
            params = stations[0] if edge in ("south", "north") else stations[1]
            for t in params[2:-2]:
                # 2e-2: the ridge least-squares' own trade at nu=12, widest
                # corner-adjacent where the two edge constraints pull (the
                # known corner-band looseness, M5's territory; measured
                # 3.6e-3 central, 8.7e-3 corner-adjacent, exact at nu=16)
                assert abs(_slope_ratio(surface, edge, t) - 2.0) < 2e-2, (edge, t)
        # interior regression anchors (M3 gate): cross-direction peak far
        # below what the unclamped plate could hold, interior means smooth
        profile = _curvature_profile(surface, 12)
        assert profile[3] < 0.25, profile  # |k_v| max
        assert profile[1] < 0.20, profile  # |k_u| max
        assert profile[2] < 0.06, profile  # |k_v| mean
