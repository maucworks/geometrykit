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


def discrete_plate_fill(
    curves: Sequence,
    nu: int = 8,
    nv: int = 8,
    tol: float = 1e-6,
    degree: int = 3,
    parametrization: str = "arclength",
):
    """A four-boundary patch whose interior is a minimising plate fill.

    The boundary is the same station grid the Coons patch uses (so the boundary
    curves are interpolated exactly and the station map is unchanged); the
    interior is *not* the bilinear blend but the solution of the thin-plate
    equation through that boundary, which is what gives an energy-minimising
    fill its even curvature instead of a flat middle with bent edges.

    The plate is simply supported (see `linalg_np.plate_fill`), which is what
    makes the fourth-order problem well posed from boundary values alone.

    Returns ``(surface, corners)`` like `discrete_coons_patch`.
    """
    from geometrykit.runtimes._npu import linalg_np

    grid, params_u, params_v, corners = discrete_coons_grid(
        curves, nu=nu, nv=nv, tol=tol, parametrization=parametrization
    )
    n_u = len(params_u)
    n_v = len(params_v)
    flat = [(point[0], point[1], point[2]) for row in grid for point in row]
    filled = linalg_np.plate_fill(flat, n_u - 1, n_v - 1)
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
