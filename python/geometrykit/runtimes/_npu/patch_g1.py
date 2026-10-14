# This file was generated with the assistance of an AI coding tool.
"""P-1: G1-constrained filling over the P-0 discrete Coons base.

The G1 constraint, made linear
------------------------------
At a boundary edge the patch's first cross-derivative ``d`` must leave the
tangent plane of the constraint surface (the ribbon). With the edge tangent
``T`` and the constraint normal ``N`` (``N . T = 0`` at the shared edge),
tangent-plane continuity -- the patch normal parallel to ``N`` -- holds
exactly when

    d . N = 0.

Only the *direction* of ``d`` is constrained; its magnitude is the patch's
own free choice. That is what makes the condition linear and cheap.

Where the rows come from
------------------------
For a clamped degree-``q`` B-spline in the cross direction the boundary
derivative is carried by the two boundary-adjacent control rows alone:

    dS/dv(u, 0) = (q / span_lo) * SUM_i N_i(u) * (P[i][1]   - P[i][0])
    dS/dv(u, 1) = (q / span_hi) * SUM_i N_i(u) * (P[i][n-2] - P[i][n-1])

so each sampled station contributes one linear row in the control net. All
four edges enter **one** system, solved as a ridge-regularised least
squares: the perturbation ``D`` minimises ``|D|^2`` subject to
``A D = -residual(P0)``. Consequences:

- the edges are solved *simultaneously* -- the interior control points
  around the corners couple them, so no sequential pass can clobber a
  neighbour (the increment-1 failure mode);
- an already-satisfying field costs nothing (zero residual, so the
  minimum-norm solution is ``D = 0``) -- the coplanar case stays planar;
- the minimum-norm tie-break keeps the patch as close to the plain Coons
  base as the constraints allow.

Corner stations (the first and last station of each edge) are excluded: there
the cross-derivative is fully determined by the two boundary curves and is
not free to move. Corner twist reconciliation is the P-2 datum protocol.

Field input
-----------
``edge_normal_fields[edge]`` is the constraint normal of that edge as a
function of the *patch* parameter:

- a callable ``t -> (nx, ny, nz)`` (unit normal),
- a sequence of ``(param, normal)`` pairs, or
- a sequence of plain normals aligned with the interior stations in order.

Station parameters come from `patch_np.coons_parametrization`; map a field
that lives in curve-parameter space with `patch_np.edge_curve_params`.

A field is most robustly obtained from the support surface itself with
`normal_field_from_support`: it projects the spine onto the support and reads
the normal there, so it does not care whether the support is one-sided
(spine on an edge) or symmetric (spine in the middle).
"""

from __future__ import annotations

from typing import Sequence

from geometrykit.geometry.primitives import Vec
from geometrykit.runtimes._npu.curve_np import _nonzero_basis
from geometrykit.runtimes._npu.patch_np import (
    coons_parametrization,
    discrete_coons_patch,
)

EDGE_NAMES = ("south", "east", "north", "west")
_EPS = 1e-12


class G1Error(RuntimeError):
    """Raised when a G1 constraint cannot be assembled from the fields given."""


def _unit(vec) -> tuple:
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    ln = (x * x + y * y + z * z) ** 0.5
    if ln < _EPS:
        raise G1Error("constraint normal is degenerate (zero length)")
    return (x / ln, y / ln, z / ln)


def _field_samples(field, params: Sequence[float], corner_stations: int = 1) -> list:
    """Normalize one edge's field into ``[(param, unit_normal), ...]``.

    All three accepted forms -- callable, ``(param, normal)`` pairs and a
    plain list aligned with the interior stations -- are sampled onto the
    *same* station grid, so equivalent fields give identical patches.

    The corner band (the outermost ``corner_stations`` on each end) is
    dropped: there the cross-derivative is pinned by the two boundary
    curves and is not free to move (see the module docstring).
    """
    lo = 1 + max(int(corner_stations), 0)
    hi = len(params) - 1 - max(int(corner_stations), 0)
    if hi <= lo:
        raise G1Error("patch has too few stations for a G1 constraint")
    slots: list = [None] * len(params)
    if callable(field):
        for k, t in enumerate(params):
            slots[k] = (t, _unit(field(t)))
    elif len(field) and isinstance(field[0], (tuple, list)) and len(field[0]) == 2:
        for p, n in field:
            if p <= _EPS or p >= 1.0 - _EPS:
                continue
            k = min(range(len(params)), key=lambda m: abs(params[m] - p))
            if slots[k] is None:
                slots[k] = (params[k], _unit(n))
    else:
        if len(field) != len(params) - 2:
            raise G1Error(
                f"constraint field holds {len(field)} normals but the patch has "
                f"{len(params) - 2} interior stations"
            )
        for k in range(1, len(params) - 1):
            slots[k] = (params[k], _unit(field[k - 1]))
    out = [slots[k] for k in range(lo, hi) if slots[k] is not None]
    if not out:
        raise G1Error("constraint field has no stations outside the corner band")
    return out


def _flat_knots(knots: Sequence[float], mults: Sequence[int]) -> list:
    flat: list = []
    for value, mult in zip(knots, mults):
        flat.extend([float(value)] * int(mult))
    return flat


def _normal_equations(rows: list, unknowns: int, ridge: float):
    """Normal equations of a scaled row system, plus the ridge weight."""
    ata = [[0.0] * unknowns for _ in range(unknowns)]
    atb = [0.0] * unknowns
    for coefs, rhs in rows:
        items = [(c, coefs[c]) for c in coefs if 0 <= c < unknowns]
        scale = max((abs(k) for _c, k in items), default=0.0)
        if scale < _EPS:
            continue
        for a, ka in items:
            atb[a] += ka * rhs / (scale * scale)
            for b, kb in items:
                ata[a][b] += ka * kb / (scale * scale)
    trace = sum(ata[i][i] for i in range(unknowns)) / max(unknowns, 1)
    lam = ridge * (trace if trace > _EPS else 1.0)
    for i in range(unknowns):
        ata[i][i] += lam
    return ata, atb


def _solve_dense(matrix: list, rhs: list) -> list:
    """Gauss-Jordan with partial pivoting on a small dense system."""
    unknowns = len(matrix)
    size = unknowns + 1
    aug = [matrix[r][:] + [rhs[r]] for r in range(unknowns)]
    for col in range(unknowns):
        piv = max(range(col, unknowns), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-14:
            raise G1Error("G1 constraint system is singular")
        aug[col], aug[piv] = aug[piv], aug[col]
        pivot_val = aug[col][col]
        for c in range(col, size):
            aug[col][c] /= pivot_val
        for r in range(unknowns):
            if r == col or aug[r][col] == 0.0:
                continue
            factor = aug[r][col]
            for c in range(col, size):
                aug[r][c] -= factor * aug[col][c]
    return [aug[r][unknowns] for r in range(unknowns)]


def _ridge_solve(rows: list, unknowns: int, ridge: float) -> list:
    """``(A^T A + ridge I) x = A^T b`` on a scalar system.

    ``rows`` are ``({col: coef}, rhs)`` pairs; returns ``[x, ...]``.
    One step of iterative refinement on the *unsquared* system recovers
    the digits the normal equations give away (the constraints are then
    met to ~1e-12 of the cross-derivative scale).
    """
    ata, atb = _normal_equations(rows, unknowns, ridge)
    solution = _solve_dense(ata, atb)
    residual_rows = []
    for coefs, rhs in rows:
        lhs = sum(coefs.get(i, 0.0) * solution[i] for i in range(unknowns))
        residual_rows.append((coefs, rhs - lhs))
    if any(abs(r) > _EPS for _c, r in residual_rows):
        ata2, atb2 = _normal_equations(residual_rows, unknowns, ridge)
        correction = _solve_dense(ata2, atb2)
        solution = [solution[i] + correction[i] for i in range(unknowns)]
    return solution


def solve_g1_constraints(
    surface,
    params_u: Sequence[float],
    params_v: Sequence[float],
    edge_normal_fields: dict,
    ridge: float = 1e-10,
    corner_stations: int = 1,
):
    """P-1: enforce the G1 tangent-plane rows on a P-0 Coons patch.

    Args:
        surface:            the P-0 discrete Coons surface (not mutated).
        params_u/params_v:  the station parameters the patch was built on.
        edge_normal_fields: ``{"south": field, ...}`` with any of
                            `EDGE_NAMES`; an absent edge stays G0.
        ridge:              relative Tikhonov weight -- ties the under-
                            determined system to the smallest patch change.

    Returns:
        A new ``Surface`` with the boundary-adjacent control rows moved into
        the constraint planes; the Coons base is untouched elsewhere.
    """
    from geometrykit.geometry.surface import Surface

    qu, qv = surface.udegree, surface.vdegree
    flat_u = _flat_knots(surface.uknots, surface.umults)
    flat_v = _flat_knots(surface.vknots, surface.vmults)
    span_v_lo = flat_v[qv + 1] - flat_v[qv]
    span_v_hi = flat_v[-qv - 1] - flat_v[-qv - 2]
    span_u_lo = flat_u[qu + 1] - flat_u[qu]
    span_u_hi = flat_u[-qu - 1] - flat_u[-qu - 2]

    net = [[(p.x, p.y, p.z) for p in row] for row in surface.control_points]
    n_u = len(net)
    n_v = len(net[0])

    # the unknown band: interior control points that carry a boundary cross
    # derivative (the boundary rows/columns are pinned by the interpolated
    # boundary curves, so corner-adjacent points on them stay put)
    band = set()
    for i in range(1, n_u - 1):
        band.add((i, 1))
        band.add((i, n_v - 2))
    for j in range(1, n_v - 1):
        band.add((1, j))
        band.add((n_u - 2, j))
    band = sorted(band)
    index = {point: 3 * k for k, point in enumerate(band)}
    if not band:
        raise G1Error("patch net too small to carry G1 constraints")

    rows = []

    def dot3(a, b) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def add_edge_rows(samples, bases, factor, inner_key, outer_key) -> None:
        for t, normal in samples:
            coefs = {}
            residual = 0.0
            for slot, weight in bases(t):
                i_slot, j_slot = slot[0], slot[1]
                inner = net[i_slot][j_slot]
                outer_i, outer_j = outer_key(i_slot, j_slot)
                outer = net[outer_i][outer_j]
                delta = (
                    inner[0] - outer[0],
                    inner[1] - outer[1],
                    inner[2] - outer[2],
                )
                coefficient = factor * weight
                residual += coefficient * dot3(delta, normal)
                base_index = index.get(inner_key(slot[0], slot[1]))
                if base_index is None:
                    continue
                for comp in range(3):
                    coefs[base_index + comp] = coefficient * normal[comp]
            rows.append((coefs, -residual))

    for edge in EDGE_NAMES:
        field = edge_normal_fields.get(edge)
        if field is None:
            continue
        if edge in ("south", "north"):
            samples = _field_samples(field, params_u, corner_stations)
            low = edge == "south"
            factor = (qv / span_v_lo) if low else (qv / span_v_hi)
            inner_row = 1 if low else n_v - 2
            outer_row = 0 if low else n_v - 1
            add_edge_rows(
                samples,
                lambda t: [
                    ((i, inner_row), n_i)
                    for i, n_i in _nonzero_basis(t, qu, flat_u).items()
                    if 0 <= i < n_u
                ],
                factor,
                lambda i, j: (i, inner_row),
                lambda i, j: (i, outer_row),
            )
        else:
            samples = _field_samples(field, params_v, corner_stations)
            low = edge == "west"
            factor = (qu / span_u_lo) if low else (qu / span_u_hi)
            inner_col = 1 if low else n_u - 2
            outer_col = 0 if low else n_u - 1
            add_edge_rows(
                samples,
                lambda t: [
                    ((inner_col, j), n_j)
                    for j, n_j in _nonzero_basis(t, qv, flat_v).items()
                    if 0 <= j < n_v
                ],
                factor,
                lambda i, j: (inner_col, j),
                lambda i, j: (outer_col, j),
            )

    if not rows:
        return surface

    unknowns = 3 * len(band)
    solution = _ridge_solve(rows, unknowns, ridge)
    out = [[Vec(p[0], p[1], p[2]) for p in row] for row in net]
    for k, (i, j) in enumerate(band):
        base_p = net[i][j]
        out[i][j] = Vec(
            base_p[0] + solution[3 * k],
            base_p[1] + solution[3 * k + 1],
            base_p[2] + solution[3 * k + 2],
        )
    return Surface(
        control_points=out,
        uknots=list(surface.uknots),
        vknots=list(surface.vknots),
        umults=list(surface.umults),
        vmults=list(surface.vmults),
        udegree=qu,
        vdegree=qv,
    )


def normal_field_from_support(support, spine, tol: float = 1e-4):
    """Constraint normals read off a support surface along its spine.

    The station is *located on the support by projection*, so the support may
    be a one-sided ribbon (spine on its boundary), a symmetric ribbon (spine
    in the middle) or a full support shell -- the caller never has to guess
    which boundary edge carries the spine. This is the contract a CAD system
    uses for a tangency support: the spine curve is what the constraint is
    attached to, and the support supplies the tangent plane there.

    Args:
        support: any `Surface` that carries the constraint plane.
        spine:   the `Curve` the constraint is attached to (a patch boundary).
        tol:     how far the spine may sit off the support before raising.

    Returns:
        A callable ``t -> (nx, ny, nz)`` of the *spine* parameter, giving the
        unit normal of the support at the projection of ``spine(t)``. Compose
        it with `patch_np.edge_curve_params` to get a field in patch
        parameter space.
    """

    def at(t: float) -> tuple:
        point = spine.point_at(t)
        projected, (u, v) = support.project(point)
        gap = abs(projected - point)
        if gap > tol:
            raise G1Error(
                f"the spine misses its support by {gap:.3e} at t={t:.4f} "
                f"(tol {tol:.1e}) -- wrong support for this curve?"
            )
        normal = support.normal_at(u, v)
        return _unit((normal.x, normal.y, normal.z))

    return at


def constrained_coons_patch(
    curves: Sequence,
    edge_normal_fields: dict,
    nu: int = 12,
    nv: int = 12,
    tol: float = 1e-6,
    degree: int = 3,
    ridge: float = 1e-10,
    corner_stations: int = 1,
):
    """P-1 entry point: G1-constrained discrete Coons patch.

    Args:
        curves:             ``[south, east, north, west]`` (drawn directions
                            auto-repaired).
        edge_normal_fields: per-edge constraint normals in patch parameter
                            space -- callable, ``(param, normal)`` pairs or
                            plain normals aligned with the interior stations
                            (see the module docstring). Absent edge = G0.
        nu/nv:              station counts for the P-0 base.
        tol:                corner coincidence tolerance.
        degree:             tensor-interpolation degree.
        ridge:              relative Tikhonov weight of the G1 solve.
        corner_stations:    stations dropped at each end of an edge (the
                            corner band, where the boundary curves pin
                            the cross-derivative).

    Returns:
        ``(surface, corners, stations)`` -- the G1-adjusted surface, the
        corner sheet, and ``(params_u, params_v)`` so callers can probe the
        patch at the very stations the constraints were placed on.
    """
    surface, corners = discrete_coons_patch(curves, nu=nu, nv=nv, tol=tol, degree=degree)
    params_u, params_v, _oriented = coons_parametrization(curves, nu=nu, nv=nv, tol=tol)
    fixed = solve_g1_constraints(
        surface,
        params_u,
        params_v,
        edge_normal_fields,
        ridge=ridge,
        corner_stations=corner_stations,
    )
    return fixed, corners, (params_u, params_v)