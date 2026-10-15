# This file was generated with the assistance of an AI coding tool.
"""Tensor-NURBS patch kernel (runtime `_npu`): P-0 discrete Coons.

The position contract for four-boundary patches: the curves are sampled
onto a common station grid (auto-oriented so the corner chain closes),
mixed by the bilinear Coons formula (corner corrections cancel the double
counting) and the resulting grid is tensor-interpolated (A9.1 per
direction) into a NURBS surface.

Guarantees:
- exact corner pass-through (validated before building — mismatch raises),
- exact boundary agreement at the station parameters,
- planar boundaries in, planar patch out (the coplanar-reflexive G1 case),
- occ-free; P-1 (the G1 cross-field rows) extends this file later.

Corner convention: ``curves = [south, east, north, west]`` — south
sampled W→E, east S→N, north W→E, west S→N; drawn directions may run
either way (auto-detected and handled).
"""

from __future__ import annotations

from typing import Sequence

from geometrykit.geometry.primitives import Vec
from geometrykit.runtimes._npu.curve_np import (
    _gauss_solve,
    _nonzero_basis,
    averaged_inner_knots,
    chord_params,
)


class PatchError(RuntimeError):
    """Raised when the boundary network cannot form a consistent patch."""


def _sample(curve, params: Sequence[float], reverse: bool) -> list:
    if reverse:
        return [curve.point_at(1.0 - t) for t in params]
    return [curve.point_at(t) for t in params]


def _arclength_stations(curve, count: int, rev: bool, samples: int = 400) -> list:
    """Parameters of *curve* at equal arc-length fractions.

    In the curve's own parameterisation (the drawn direction is accounted for),
    one per patch station, so the grid samples the curve directly and a
    constraint field maps back onto it.
    """

    def forward(t: float):
        return curve.point_at(1.0 - t) if rev else curve.point_at(t)

    ts = [i / samples for i in range(samples + 1)]
    points = [forward(t) for t in ts]
    cumulative = [0.0]
    for i in range(1, len(points)):
        cumulative.append(cumulative[-1] + abs(points[i] - points[i - 1]))
    total = cumulative[-1]
    stations = []
    cursor = 0
    for k in range(count + 1):
        target = total * k / count
        while cursor < len(cumulative) - 2 and cumulative[cursor + 1] < target:
            cursor += 1
        span = cumulative[cursor + 1] - cumulative[cursor]
        fraction = 0.0 if span <= 1e-15 else (target - cumulative[cursor]) / span
        t = ts[cursor] + fraction * (ts[cursor + 1] - ts[cursor])
        stations.append(1.0 - t if rev else t)
    return stations


def _match_between(a: tuple, b: tuple, tol: float) -> bool:
    return abs(a - b) < tol


def orient_cycle(curves: Sequence, tol: float = 1e-6) -> dict:
    """Orient the four boundary curves onto the closed corner cycle.

    Convention (after resolution): south curves W→E, east S→N, north W→E,
    west S→N. Returns ``{name: (curve, reversed_flag)}`` and validates the
    four corner coincidences.
    """
    if len(curves) != 4:
        raise PatchError("patch: exactly four boundary curves are required")
    south = curves[0]
    c00 = south.point_at(0.0)
    c10 = south.point_at(1.0)
    rest = list(curves[1:])

    # east: touches c10 (either end)
    east = None
    east_rev = False
    for i, c in enumerate(rest):
        s, e = c.point_at(0.0), c.point_at(1.0)
        if abs(s - c10) < tol:
            east, east_rev = c, False
            rest.pop(i)
            break
        if abs(e - c10) < tol:
            east, east_rev = c, True
            rest.pop(i)
            break
    if east is None:
        raise PatchError("patch: no boundary curve touches the south-east corner")
    # c11 = east's FAR end (the north-east corner), not its matched start
    c11 = east.point_at(1.0) if not east_rev else east.point_at(0.0)

    # north: sampled W→E — must touch c01 on one end and c11 on the other
    north = None
    north_rev = False
    c01 = None
    # candidates touching c11 on exactly one end; the other end defines c01
    for i, c in enumerate(rest):
        s, e = c.point_at(0.0), c.point_at(1.0)
        if abs(s - c11) < tol and abs(e - c11) > tol:
            north, north_rev = c, True  # drawn NE→NW: sampled W→E reversed
            c01 = e
            rest.pop(i)
            break
        if abs(e - c11) < tol and abs(s - c11) > tol:
            north, north_rev = c, False  # drawn W→E as needed
            c01 = s
            rest.pop(i)
            break
    if north is None:
        raise PatchError("patch: no boundary curve touches the north-east corner")

    # west: sampled S→N — touched c00 (S) and c01 (N) in either drawn order.
    west = None
    west_rev = False
    for i, c in enumerate(rest):
        s, e = c.point_at(0.0), c.point_at(1.0)
        if abs(e - c01) < tol and abs(s - c00) < tol:
            west, west_rev = c, False  # drawn S→N
            rest.pop(i)
            break
        if abs(s - c01) < tol and abs(e - c00) < tol:
            west, west_rev = c, True  # drawn N→S
            rest.pop(i)
            break
    if west is None:
        raise PatchError("patch: the west boundary is missing")

    # corner coincidence validation (the cycle must close)
    C11_actual = north.point_at(0.0 if north_rev else 1.0)
    if abs(C11_actual - c11) > tol:
        raise PatchError(f"patch: corner C11 (north/east) mismatch {abs(C11_actual - c11):.3e}")
    C00_actual = west.point_at(1.0 if west_rev else 0.0)
    if abs(C00_actual - c00) > tol:
        raise PatchError(f"patch: corner C00 (west/south) mismatch {abs(C00_actual - c00):.3e}")
    return {
        "south": (south, False),
        "east": (east, east_rev),
        "north": (north, north_rev),
        "west": (west, west_rev),
        "corners": {"C00": c00, "C10": c10, "C01": c01, "C11": c11},
    }


def _comp_corn(c00, c10, c01, c11, a: float, b: float) -> tuple:
    return [
        (1.0 - a) * (1.0 - b) * c00[k] + a * (1.0 - b) * c10[k]
        + (1.0 - a) * b * c01[k] + a * b * c11[k]
        for k in range(3)
    ]


def coons_parametrization(
    curves: Sequence,
    nu: int = 8,
    nv: int = 8,
    tol: float = 1e-6,
    parametrization: str = "chord",
):
    """The station parameters the Coons patch is built on.

    Returns ``(params_u, params_v, oriented)``. ``oriented[edge]`` is
    ``(curve, stations)``: the curve, and for every patch station the parameter
    at which to sample it -- so a constraint field maps from patch parameter
    space back onto the curve with `edge_curve_params`, and the invariant
    ``surface.point_at(params[i], edge) == curve.point_at(stations[i])`` holds.

    ``parametrization`` decides how the stations are spread:

    - ``"chord"`` (default): the curve is sampled at uniform parameter, so the
      patch parameters are the chord lengths between those samples and the
      patch parameterisation *is* the curve's own. The spacing then follows the
      curve's speed, which for a Bezier with long end handles means stations
      bunched in the middle and wide at the ends.
    - ``"arclength"``: equal arc length, so the stations are evenly spread in
      space and the parameters are uniform. Paired with uniform knots (see
      `discrete_coons_patch`), the Greville abscissae are uniform too, which is
      what keeps the control net evenly spaced and the boundary derivative
      scaling constant.
    """
    oriented = orient_cycle(curves, tol=tol)
    south_curve, south_rev = oriented["south"]
    east_curve, east_rev = oriented["east"]
    if parametrization == "arclength":
        params_u = [i / nu for i in range(nu + 1)]
        params_v = [j / nv for j in range(nv + 1)]
    elif parametrization == "chord":
        south_fwd = _sample(south_curve, [i / nu for i in range(nu + 1)], south_rev)
        params_u = chord_params(south_fwd)
        east_fwd = _sample(east_curve, [j / nv for j in range(nv + 1)], east_rev)
        params_v = chord_params(east_fwd)
    else:
        raise PatchError(
            f"unknown parametrization {parametrization!r}: use 'chord' or 'arclength'"
        )

    sheet = {}
    for name in ("south", "east", "north", "west"):
        curve, rev = oriented[name]
        count = nu if name in ("south", "north") else nv
        params = params_u if name in ("south", "north") else params_v
        if parametrization == "arclength":
            sheet[name] = (curve, _arclength_stations(curve, count, rev))
        else:
            sheet[name] = (curve, [1.0 - p if rev else p for p in params])
    sheet["corners"] = oriented["corners"]
    return params_u, params_v, sheet


def edge_curve_params(oriented: dict, edge: str, patch_params: Sequence[float]) -> list:
    """Map patch-space station params onto one edge's own curve parameter.

    ``edge_curve_params(oriented, "north", params_u)[i]`` is the parameter on
    the north curve at which the i-th u-station sits. The station map already
    accounts for a drawn direction that runs opposite to the patch's sampled
    one, and for the parameterisation choice.
    """
    entry = oriented[edge]
    if len(entry) == 2:
        _curve, stations = entry
        return list(stations)
    _curve, rev = entry
    if rev:
        return [1.0 - p for p in patch_params]
    return list(patch_params)


def discrete_coons_grid(
    curves: Sequence,
    nu: int = 8,
    nv: int = 8,
    tol: float = 1e-6,
    parametrization: str = "chord",
):
    """Sample the four boundaries onto the discrete Coons grid.

    Returns ``(grid, params_u, params_v, corners)`` — grid[i][j] = (x, y, z).
    """
    params_u, params_v, orient = coons_parametrization(
        curves, nu=nu, nv=nv, tol=tol, parametrization=parametrization
    )
    south_curve, south_rev = orient["south"]
    east_curve, east_rev = orient["east"]
    north_curve, north_rev = orient["north"]
    west_curve, west_rev = orient["west"]

    Su = [south_curve.point_at(t) for t in orient["south"][1]]
    Sn = [north_curve.point_at(t) for t in orient["north"][1]]
    Se = [east_curve.point_at(t) for t in orient["east"][1]]
    Sw = [west_curve.point_at(t) for t in orient["west"][1]]

    corners = orient["corners"]

    grid = []
    for i in range(nu + 1):
        a = params_u[i]
        row = []
        for j in range(nv + 1):
            b = params_v[j]
            sa, sn, sw, se = Su[i], Sn[i], Sw[j], Se[j]
            bilinear = _comp_corn(
                tuple(corners["C00"]), tuple(corners["C10"]),
                tuple(corners["C01"]), tuple(corners["C11"]), a, b,
            )
            v = [
                (1.0 - b) * sa[0] + b * sn[0] + (1.0 - a) * sw[0] + a * se[0] - bilinear[0],
                (1.0 - b) * sa[1] + b * sn[1] + (1.0 - a) * sw[1] + a * se[1] - bilinear[1],
                (1.0 - b) * sa[2] + b * sn[2] + (1.0 - a) * sw[2] + a * se[2] - bilinear[2],
            ]
            row.append(v)
        grid.append(row)
    return grid, params_u, params_v, corners


def _interp_1d_vec(values: list, params: list, q: int, uniform_knots: bool = False):
    """A9.1 interpolation of Vec-value records along one direction.

    Returns ``(cps, knots, mults)`` — cps as plain component tuples.
    """
    unknowns = len(values)
    q = max(1, min(q, unknowns - 1))
    inner = unknowns - q - 1
    flat: list = [0.0] * (q + 1)
    if inner > 0:
        # window_offset=0 is the placement this interpolation is solvable
        # with: it keeps the last data parameters covered by the averaging
        # windows. Piegl's A9.1 rule (window_offset=-1) leaves the last span
        # wide and the system ill-conditioned for chord parameters -- the
        # solved poles then blow up. The consequence of offset=0 is a first
        # span about twice the first station gap, which is what crowds the
        # control net away from the boundary (see the parameterisation note in
        # the module docstring).
        if uniform_knots:
            # uniform clamped knots, spacing 1/(inner+1): pairs with uniform
            # parameters so the Greville abscissae are uniform too
            flat += [j / (inner + 1) for j in range(1, inner + 1)]
        else:
            flat += averaged_inner_knots(params[: inner + q], q, inner, window_offset=0)
    flat += [1.0] * (q + 1)

    rows = []
    for k, val in enumerate(values):
        Nu = _nonzero_basis(params[k], q, flat)
        rows.append((Nu, val))
    sol = _gauss_solve(rows, unknowns, dims=3)
    cps = [(sol[i][0], sol[i][1], sol[i][2]) for i in range(unknowns)]
    knots, mults = [], []
    for val in flat:
        if knots and abs(val - knots[-1]) < 1e-9:
            mults[-1] += 1
        else:
            knots.append(val)
            mults.append(1)
    return cps, knots, mults


def _tensor_interpolate(
    grid: list,
    params_u: list,
    params_v: list,
    degree: int,
    uniform_knots: bool,
):
    """Interpolate a station grid into a tensor-product NURBS `Surface`."""
    from geometrykit.geometry.surface import Surface

    n_u = len(params_u)
    n_v = len(params_v)
    qu = max(1, min(degree, n_u - 1))
    qv = max(1, min(degree, n_v - 1))

    per_col = []
    for j in range(n_v):
        vals = [tuple(grid[i][j]) for i in range(n_u)]
        cps, knots_u, mults_u = _interp_1d_vec(
            vals, params_u[:n_u], qu, uniform_knots=uniform_knots
        )
        per_col.append((cps, knots_u, mults_u))
    knots_u, mults_u = per_col[0][1], per_col[0][2]
    n_u_slots = len(per_col[0][0])

    net_rows: list = []
    knots_v = mults_v = None
    for i in range(n_u_slots):
        col = [per_col[j][0][i] for j in range(n_v)]
        cps_i, kv, mv = _interp_1d_vec(
            col, params_v[:n_v], qv, uniform_knots=uniform_knots
        )
        net_rows.append([Vec(*p) for p in cps_i])
        knots_v, mults_v = kv, mv

    return Surface(
        control_points=net_rows,
        uknots=knots_u,
        vknots=knots_v,
        umults=mults_u,
        vmults=mults_v,
        udegree=qu,
        vdegree=qv,
    )


def discrete_coons_patch(
    curves: Sequence,
    nu: int = 8,
    nv: int = 8,
    tol: float = 1e-6,
    degree: int = 3,
    parametrization: str = "chord",
):
    """P-0: four-boundary discrete Coons patch as a tensor NURBS `Surface`.

    Args:
        curves:  `[south, east, north, west]` (drawn directions auto-repaired).
        nu/nv:   station counts along the u/v directions (degree-3 sampling grid).
        tol:     corner coincidence tolerance.
        degree:  tensor-interpolation degree (clamped per direction).

    Returns:
        ``(surface, corners)`` — the constructed Surface + the corner sheet.
    """

    grid, params_u, params_v, corners = discrete_coons_grid(
        curves, nu=nu, nv=nv, tol=tol, parametrization=parametrization
    )
    surface = _tensor_interpolate(
        grid,
        params_u,
        params_v,
        degree,
        uniform_knots=parametrization == "arclength",
    )
    return surface, {k: Vec(*v) for k, v in corners.items()}


def _extension_point(
    curve, side: str, arc_offset: float
) -> tuple:
    """Extend a curve beyond an endpoint along its own tangent and curvature.

    The cubic Taylor of the curve around the endpoint, evaluated at the arc
    offset *outward*: p(-s) = p(0) - s*T0 + s^2/2 * dT/ds with T0 the unit
    end tangent and dT/ds the curvature vector from a backward finite
    difference. This is belt data (the loose fill's outer ring, which gets
    trimmed away), so second-order fidelity is the point of the exercise.
    """
    if side == "start":
        t0, t1 = 0.0, 1e-3
        sign = -1.0
    else:
        t0, t1 = 1.0, 1.0 - 1e-3
        sign = 1.0
    p0 = curve.point_at(t0)
    tan0 = curve.tangent_at(t0)
    tan1 = curve.tangent_at(t1)
    n0 = (tan0.x**2 + tan0.y**2 + tan0.z**2) ** 0.5
    n1 = (tan1.x**2 + tan1.y**2 + tan1.z**2) ** 0.5
    t_hat = (tan0.x / n0, tan0.y / n0, tan0.z / n0)
    t_next = (tan1.x / n1, tan1.y / n1, tan1.z / n1)
    p1 = curve.point_at(t1)
    ds = (abs(p1.x - p0.x) + abs(p1.y - p0.y) + abs(p1.z - p0.z)) or 1e-9
    curvature = tuple(sign * (t_next[k] - t_hat[k]) / ds for k in range(3))
    x = p0.x + sign * arc_offset * t_hat[0] + 0.5 * arc_offset**2 * curvature[0]
    y = p0.y + sign * arc_offset * t_hat[1] + 0.5 * arc_offset**2 * curvature[1]
    z = p0.z + sign * arc_offset * t_hat[2] + 0.5 * arc_offset**2 * curvature[2]
    return (x, y, z)


def discrete_plate_fill(
    curves: Sequence,
    nu: int = 8,
    nv: int = 8,
    tol: float = 1e-6,
    degree: int = 3,
    parametrization: str = "arclength",
    slopes: dict | None = None,
    curvatures: dict | None = None,
    steps: tuple | None = None,
    loose: int = 0,
):
    """A four-boundary patch whose interior is a minimising plate fill.

    The boundary is the same station grid the Coons patch uses (so the
    boundary curves are interpolated exactly and the station map is
    unchanged); the interior is *not* the bilinear blend but the solution
    of the thin-plate equation through that boundary, which is what gives
    an energy-minimising fill its even curvature instead of a flat middle
    with bent edges.

    The plate is simply supported by default (see `linalg_np.plate_fill`),
    which is what makes the fourth-order problem well posed from boundary
    values alone. ``slopes`` upgrades a side to the clamped closure -- the
    support's cross-derivative is then a boundary condition of the solve
    itself, which is what gives an energy-minimising fill its stiffness
    without post-hoc corrections. ``curvatures`` (per side, needing slopes
    on that side) pins the first ring inward to the Hermite value -- the
    G2 condition inside the solve. ``steps`` is the grid spacing
    ``(du, dv)``.

    With ``loose`` = e > 0 the solve runs over a grid *enlarged* by e
    stations on every side -- this library's version of OCC's plate
    construction, where the fill surface runs *through* the edges and is
    trimmed afterwards: the original edge lines become exact interior
    Dirichlet lines, the slope conditions become interior derivative lines
    (claims: one station outward expressed through the inward neighbour),
    and the extension belt's outer ring comes from tangent+curvature
    continuation of the boundary curves. What comes back is the window
    rows e..e+nu-1 re-interpolated with the original station parameters:
    the boundary is exact as ever, and the plate's bending is free to
    spend some of itself on the belt instead of the first visible span.

    Returns ``(surface, corners)`` like `discrete_coons_patch`.
    """
    from geometrykit.runtimes._npu import linalg_np

    grid, params_u, params_v, corners = discrete_coons_grid(
        curves, nu=nu, nv=nv, tol=tol, parametrization=parametrization
    )
    n_u = len(params_u)
    n_v = len(params_v)
    if steps is None:
        steps = (params_u[1] - params_u[0], params_v[1] - params_v[0])
    if loose <= 0:
        flat = [(point[0], point[1], point[2]) for row in grid for point in row]
        filled = linalg_np.plate_fill(
            flat, n_u - 1, n_v - 1, slopes=slopes, curvatures=curvatures, steps=steps
        )
        solved = [
            [filled[i * n_v + j] for j in range(n_v)] for i in range(n_u)
        ]
        surface = _tensor_interpolate(
            solved,
            params_u,
            params_v,
            degree,
            uniform_knots=parametrization == "arclength",
        )
        return surface, {k: Vec(*v) for k, v in corners.items()}
    return _loose_plate_fill(
        curves, corners, params_u, params_v, steps, loose,
        degree, parametrization, slopes, curvatures,
    )


_ORIGINAL_EDGES = ("south", "east", "north", "west")


def _lin_extend_line(points: list, count: int, which: str) -> list:
    """Linear continuation of a polyline beyond one end, per index step.

    Used for the corner cells of the loose fill's outer ring, where no
    cross-slope data exists: each corner cell continues the ring's own
    local tangent. One step per ring index cell.
    """
    if not points:
        raise ValueError("cannot extend an empty belt line")
    if which == "start":
        step = _sub3(_at(points, 1), _at(points, 0))
        return [_sub3(_at(points, 0), _cust(step, k)) for k in range(count, 0, -1)]
    step = _sub3(_at(points, -1), _at(points, -2))
    return [_add3(_at(points, -1), _cust(step, k)) for k in range(1, count + 1)]


def _at(points, k):
    return points[k]


def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add3(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _cust(step, k):
    return tuple(k * c for c in step)


def _loose_plate_fill(
    curves, corners, params_u, params_v, steps, e,
    degree, parametrization, slopes, curvatures,
):
    """The loose construction: solve on the enlarged grid, window it.

    The enlarged grid places the original edges at margin e. The outer
    ring continues each edge *outward in its cross direction* along the
    support's own cross profile -- first order in the folded slope data,
    with a second-order term from the curvature data when it is supplied;
    the corner cells (no cross data) continue each ring's local tangent.
    The original edge lines are exact interior Dirichlet data; the slope
    conditions are interior derivative lines expressed as claims (the
    outward belt station stated through the inward neighbour). The window
    re-interpolates the original station parameters, so the boundary is as
    exact as ever and the plate's bending is free to spend part of itself
    on the belt instead of the first visible span.
    """
    from geometrykit.runtimes._npu import linalg_np

    du, dv = steps
    n_u = len(params_u)
    n_v = len(params_v)
    n_xu = n_u + 2 * e
    n_xv = n_v + 2 * e
    stride = n_xv
    oriented = coons_parametrization(
        curves, nu=n_u - 1, nv=n_v - 1, parametrization=parametrization
    )[2]

    lines: dict = {}
    for edge in _ORIGINAL_EDGES:
        k = _ORIGINAL_EDGES.index(edge)
        params_line = params_u if edge in ("south", "north") else params_v
        curve_params = edge_curve_params(oriented, edge, params_line)
        points = [
            (p.x, p.y, p.z)
            for p in (curves[k].point_at(t) for t in curve_params)
        ]
        lines[edge] = points

    def _ring(edge, line, factor, step):
        """One outer ring line: the edge offset along the cross slope.

        ``factor`` is the outward sign in index units: −1 on west/south,
        +1 on east/north. Folded slope/curvature arrays are per station,
        and the offset is ``e`` cells of ``e`` per unit index, so the cell
        at distance k sits at factor * (k * step) per unit of the cross
        parameter -- the whole ring is at k = e cells.
        """
        out = []
        for s, point in enumerate(line):
            slope = slopes[edge][s] if slopes is not None else (0.0, 0.0, 0.0)
            kappa = curvatures[edge][s] if curvatures is not None else (0.0, 0.0, 0.0)
            # the ring cell sits e cells away from the edge: offset in
            # cross parameter units = factor * e * step (folded slope and
            # curvature arrays are derivatives per increasing-index unit)
            offset = factor * e * step
            out.append(
                tuple(
                    point[c]
                    + offset * slope[c]
                    + 0.5 * offset * offset * kappa[c]
                    for c in range(3)
                )
            )
        return out

    south_ring = _ring("south", lines["south"], -1.0, dv)
    north_ring = _ring("north", lines["north"], 1.0, dv)
    west_ring = _ring("west", lines["west"], -1.0, du)
    east_ring = _ring("east", lines["east"], 1.0, du)

    values: list = [(0.0, 0.0, 0.0)] * (n_xu * n_xv)
    for s, point in enumerate(south_ring):
        values[(e + s) * stride] = point
    for s, point in enumerate(north_ring):
        values[(e + s) * stride + (n_xv - 1)] = point
    for s, point in enumerate(west_ring):
        values[e + s] = point
    for s, point in enumerate(east_ring):
        values[(n_xu - 1) * stride + e + s] = point

    def _place_belt(belt_line, cells, mode):
        for k, point in enumerate(belt_line):
            coords = cells[k]
            values[coords[0] * stride + coords[1]] = tuple(point)

    # corner cells: linear continuation of each ring along its own tangent
    south_cells = [(i, 0) for i in range(e)]
    _place_belt(_lin_extend_line(south_ring, e, "start"), south_cells, "s")
    south_cells_hi = [(e + n_u - 1 + k + 1, 0) for k in range(e)]
    _place_belt(_lin_extend_line(south_ring, e, "end"), south_cells_hi, "s")
    north_cells_lo = [(i, n_xv - 1) for i in range(e)]
    _place_belt(_lin_extend_line(north_ring, e, "start"), north_cells_lo, "n")
    north_cells_hi = [(e + n_u - 1 + k + 1, n_xv - 1) for k in range(e)]
    _place_belt(_lin_extend_line(north_ring, e, "end"), north_cells_hi, "n")
    west_cells_lo = [(0, j) for j in range(e)]
    _place_belt(_lin_extend_line(west_ring, e, "start"), west_cells_lo, "w")
    west_cells_hi = [(0, e + n_v - 1 + k + 1) for k in range(e)]
    _place_belt(_lin_extend_line(west_ring, e, "end"), west_cells_hi, "w")
    east_cells_lo = [(n_xu - 1, j) for j in range(e)]
    _place_belt(_lin_extend_line(east_ring, e, "start"), east_cells_lo, "e")
    east_cells_hi = [(n_xu - 1, e + n_v - 1 + k + 1) for k in range(e)]
    _place_belt(_lin_extend_line(east_ring, e, "end"), east_cells_hi, "e")

    # interior initial guess (overwritten by the solve): bilinear blend
    for i in range(1, n_xu - 1):
        for j in range(1, n_xv - 1):
            b = j / (n_xv - 1)
            left = values[i * stride][:]
            right = values[i * stride + (n_xv - 1)][:]
            values[i * stride + j] = (
                (1 - b) * left[0] + b * right[0],
                (1 - b) * left[1] + b * right[1],
                (1 - b) * left[2] + b * right[2],
            )

    fixed: set = set()
    for edge, (i_line, j_offset) in (
        ("south", (0, e)), ("north", (0, e + n_v - 1)),
    ):
        for s, point in enumerate(lines[edge]):
            values[(e + s) * stride + j_offset] = tuple(point)
            fixed.add((e + s, j_offset))
    for edge, (j_line, i_offset) in (("west", (e, 0)), ("east", (e, e + n_v - 1))):
        for s, point in enumerate(lines[edge]):
            if edge == "west":
                values[j_line * stride + (e + s)] = tuple(point)
                fixed.add((j_line, e + s))
            else:
                values[(e + n_u - 1) * stride + (e + s)] = tuple(point)
                fixed.add((e + n_u - 1, e + s))

    claims: dict = {}
    if slopes is not None:
        for edge, vectors in slopes.items():
            if edge == "south":
                for s, vector in enumerate(vectors):
                    claims[(e + s, e - 1)] = (
                        {(e + s, e + 1): 1.0},
                        tuple(-2.0 * dv * c for c in vector),
                    )
            elif edge == "north":
                for s, vector in enumerate(vectors):
                    claims[(e + s, e + n_v)] = (
                        {(e + s, e + n_v - 2): 1.0},
                        tuple(2.0 * dv * c for c in vector),
                    )
            elif edge == "west":
                for s, vector in enumerate(vectors):
                    claims[(e - 1, e + s)] = (
                        {(e + 1, e + s): 1.0},
                        tuple(-2.0 * du * c for c in vector),
                    )
            else:
                for s, vector in enumerate(vectors):
                    claims[(e + n_u, e + s)] = (
                        {(e + n_u - 2, e + s): 1.0},
                        tuple(2.0 * du * c for c in vector),
                    )

    flat = [values[i * stride + j] for i in range(n_xu) for j in range(n_xv)]
    filled = linalg_np.plate_fill(
        flat, n_xu - 1, n_xv - 1, claims=claims, fixed=fixed, steps=steps
    )

    solved = [
        [filled[(e + i) * stride + (e + j)] for j in range(n_v)]
        for i in range(n_u)
    ]
    surface = _tensor_interpolate(
        solved, params_u, params_v, degree,
        uniform_knots=parametrization == "arclength",
    )
    return surface, corners
