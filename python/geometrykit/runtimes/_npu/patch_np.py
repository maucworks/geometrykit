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


def coons_parametrization(curves: Sequence, nu: int = 8, nv: int = 8, tol: float = 1e-6):
    """The station parameters the Coons patch is built on.

    Returns ``(params_u, params_v, oriented)`` -- the chord parameters on
    the oriented south (u) and east (v) boundaries, plus the orientation
    sheet from `orient_cycle`. The u/v stations double as the sampling
    stations of the north/west boundaries, so a caller holding a field in
    *curve* parameter space maps it into patch space with
    `edge_curve_params`.
    """
    oriented = orient_cycle(curves, tol=tol)
    south_curve, south_rev = oriented["south"]
    east_curve, east_rev = oriented["east"]
    south_fwd = _sample(south_curve, [i / nu for i in range(nu + 1)], south_rev)
    params_u = chord_params(south_fwd)
    east_fwd = _sample(east_curve, [j / nv for j in range(nv + 1)], east_rev)
    params_v = chord_params(east_fwd)
    return params_u, params_v, oriented


def edge_curve_params(oriented: dict, edge: str, patch_params: Sequence[float]) -> list:
    """Map patch-space station params onto one edge's own curve parameter.

    Returns the parameter on the edge curve at which each patch-space
    station sits -- accounting for a drawn direction that runs opposite
    to the patch's sampled direction.
    """
    _curve, rev = oriented[edge]
    if rev:
        return [1.0 - p for p in patch_params]
    return list(patch_params)


def discrete_coons_grid(curves: Sequence, nu: int = 8, nv: int = 8, tol: float = 1e-6):
    """Sample the four boundaries onto the discrete Coons grid.

    Returns ``(grid, params_u, params_v, corners)`` — grid[i][j] = (x, y, z).
    """
    params_u, params_v, orient = coons_parametrization(curves, nu=nu, nv=nv, tol=tol)
    south_curve, south_rev = orient["south"]
    east_curve, east_rev = orient["east"]
    north_curve, north_rev = orient["north"]
    west_curve, west_rev = orient["west"]

    Su = _sample(south_curve, params_u, south_rev)
    Sn = _sample(north_curve, params_u, north_rev)
    Se = _sample(east_curve, params_v, east_rev)
    Sw = _sample(west_curve, params_v, west_rev)

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


def _interp_1d_vec(values: list, params: list, q: int):
    """A9.1 interpolation of Vec-value records along one direction.

    Returns ``(cps, knots, mults)`` — cps as plain component tuples.
    """
    unknowns = len(values)
    q = max(1, min(q, unknowns - 1))
    inner = unknowns - q - 1
    flat: list = [0.0] * (q + 1)
    if inner > 0:
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


def discrete_coons_patch(
    curves: Sequence,
    nu: int = 8,
    nv: int = 8,
    tol: float = 1e-6,
    degree: int = 3,
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
    from geometrykit.geometry.surface import Surface

    grid, params_u, params_v, corners = discrete_coons_grid(curves, nu=nu, nv=nv, tol=tol)
    n_u = len(params_u)
    n_v = len(params_v)
    qu = max(1, min(degree, nu))
    qv = max(1, min(degree, nv))

    # step 1: interpolate each grid COLUMN j along u — intermediate columns
    per_col = []
    for j in range(n_v):
        vals = [tuple(grid[i][j]) for i in range(n_u)]
        cps, knots_u, mults_u = _interp_1d_vec(vals, params_u[: n_u], qu)
        per_col.append((cps, knots_u, mults_u))
    knots_u, mults_u = per_col[0][1], per_col[0][2]
    n_u_slots = len(per_col[0][0])

    # step 2: interpolate each u-slot's column along v — the final net rows
    net_rows: list = []
    knots_v = mults_v = None
    for i in range(n_u_slots):
        col = [per_col[j][0][i] for j in range(n_v)]
        cps_i, kv, mv = _interp_1d_vec(col, params_v[: n_v], qv)
        net_rows.append([Vec(*p) for p in cps_i])
        knots_v, mults_v = kv, mv

    surface = Surface(
        control_points=net_rows,
        uknots=knots_u,
        vknots=knots_v,
        umults=mults_u,
        vmults=mults_v,
        udegree=qu,
        vdegree=qv,
    )
    return surface, {k: Vec(*v) for k, v in corners.items()}
