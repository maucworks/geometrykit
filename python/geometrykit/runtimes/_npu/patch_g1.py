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
own free choice. That is what makes the condition linear and cheap -- and it
is also the one thing G1 leaves unspecified: how stiffly the patch leaves the
edge. `solve_g1_constraints` takes an optional ``tangency`` factor for that
magnitude (see below); without it the magnitude is whatever the
minimum-norm solve leaves behind, which is typically much softer than a CAD
system's tangency support.

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
    discrete_plate_fill,
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


def _slope_of(field, t: float):
    """The support slope carried by a field spec, or None if it has none."""
    if isinstance(field, dict):
        slope = field.get("slope")
        return None if slope is None else slope(t)
    slope = getattr(field, "slope", None)
    return None if slope is None else slope(t)


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
    if isinstance(field, dict):
        field = field["normal"]
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
    for row in rows:
        coefs, rhs = row[0], row[1]
        weight = row[2] if len(row) > 2 else 1.0
        items = [(c, coefs[c]) for c in coefs if 0 <= c < unknowns]
        scale = max((abs(k) for _c, k in items), default=0.0)
        if scale < _EPS:
            continue
        for a, ka in items:
            atb[a] += ka * rhs * weight / (scale * scale)
            for b, kb in items:
                ata[a][b] += ka * kb * weight / (scale * scale)
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
    for row in rows:
        coefs, rhs = row[0], row[1]
        weight = row[2] if len(row) > 2 else 1.0
        lhs = sum(coefs.get(i, 0.0) * solution[i] for i in range(unknowns))
        residual_rows.append((coefs, rhs - lhs, weight))
    if any(abs(row[1]) > _EPS for row in residual_rows):
        ata2, atb2 = _normal_equations(residual_rows, unknowns, ridge)
        correction = _solve_dense(ata2, atb2)
        solution = [solution[i] + correction[i] for i in range(unknowns)]
    return solution


def _second_deriv_weights(flat: list, degree: int) -> tuple:
    """Weights giving ``S''`` at the two clamped ends from the poles.

    Returns ``(low, high)``: ``{pole_index: weight}`` for ``S''(0)`` and
    ``S''(1)``. Only three poles contribute at each end.
    """
    if degree < 2:
        return {}, {}
    n = len(flat) - degree - 1  # number of poles
    span_lo = flat[degree + 1] - flat[degree]
    if span_lo <= 0.0:
        raise G1Error("degenerate knot span at the start of the net")
    # low end: poles 0, 1, 2
    first = flat[degree + 1]
    second = flat[degree + 2]
    low = {
        0: 6.0 / (span_lo * span_lo),
        1: -6.0 * (1.0 / (span_lo * second) + 1.0 / (span_lo * span_lo)),
        2: 6.0 / (span_lo * second),
    }
    del first
    # high end: the mirror, poles n-1, n-2, n-3
    span_hi = flat[-degree - 1] - flat[-degree - 2]
    if span_hi <= 0.0:
        raise G1Error("degenerate knot span at the end of the net")
    back_first = 1.0 - flat[-degree - 2]
    back_second = 1.0 - flat[-degree - 3]
    high = {
        n - 1: 6.0 / (span_hi * span_hi),
        n - 2: -6.0 * (1.0 / (span_hi * back_second) + 1.0 / (span_hi * span_hi)),
        n - 3: 6.0 / (span_hi * back_second),
    }
    del back_first
    return low, high


def _second_deriv_weights(flat: list, degree: int) -> tuple:
    """Weights giving ``S''`` at the two clamped ends from the poles.

    Returns ``(low, high)``: ``{pole_index: weight}`` for ``S''(0)`` and
    ``S''(1)``. Only three poles contribute at each end.
    """
    if degree < 2:
        return {}, {}
    n = len(flat) - degree - 1  # number of poles
    span_lo = flat[degree + 1] - flat[degree]
    if span_lo <= 0.0:
        raise G1Error("degenerate knot span at the start of the net")
    # low end: poles 0, 1, 2
    first = flat[degree + 1]
    second = flat[degree + 2]
    low = {
        0: 6.0 / (span_lo * span_lo),
        1: -6.0 * (1.0 / (span_lo * second) + 1.0 / (span_lo * span_lo)),
        2: 6.0 / (span_lo * second),
    }
    del first
    # high end: the mirror, poles n-1, n-2, n-3
    span_hi = flat[-degree - 1] - flat[-degree - 2]
    if span_hi <= 0.0:
        raise G1Error("degenerate knot span at the end of the net")
    back_first = 1.0 - flat[-degree - 2]
    back_second = 1.0 - flat[-degree - 3]
    high = {
        n - 1: 6.0 / (span_hi * span_hi),
        n - 2: -6.0 * (1.0 / (span_hi * back_second) + 1.0 / (span_hi * span_hi)),
        n - 3: 6.0 / (span_hi * back_second),
    }
    del back_first
    return low, high


def _energy_rows(net: list, index: dict, n_u: int, n_v: int, weight: float) -> list:
    """Discrete plate energy: second differences along u, v and the mixed one.

    ``|S_uu|^2 + 2|S_uv|^2 + |S_vv|^2`` in finite differences, one row per
    component per stencil, aimed at zero. Neighbours that are not movable (the
    boundary poles) contribute to the right-hand side.
    """
    rows = []
    for (i, j) in sorted(index):
        for axis in ("u", "v"):
            first = (i - 1, j) if axis == "u" else (i, j - 1)
            second = (i + 1, j) if axis == "u" else (i, j + 1)
            if not (0 <= first[0] < n_u and 0 <= first[1] < n_v):
                continue
            if not (0 <= second[0] < n_u and 0 <= second[1] < n_v):
                continue
            for comp in range(3):
                coefs = {index[(i, j)] + comp: -2.0 * weight}
                fixed = 0.0
                for neighbour in (first, second):
                    slot = index.get(neighbour)
                    if slot is None:
                        fixed += weight * net[neighbour[0]][neighbour[1]][comp]
                    else:
                        coefs[slot + comp] = weight
                rows.append((coefs, -fixed, 1.0))
        # the mixed second difference: P++ - P+- - P-+ + P--
        if not (0 <= i - 1 < n_u and 0 <= i + 1 < n_u):
            continue
        if not (0 <= j - 1 < n_v and 0 <= j + 1 < n_v):
            continue
        corners = (
            ((i + 1, j + 1), 1.0),
            ((i + 1, j - 1), -1.0),
            ((i - 1, j + 1), -1.0),
            ((i - 1, j - 1), 1.0),
        )
        for comp in range(3):
            coefs = {}
            fixed = 0.0
            for (point, sign) in corners:
                coefficient = sign * weight * 2.0
                slot = index.get(point)
                if slot is None:
                    fixed -= coefficient * net[point[0]][point[1]][comp]
                else:
                    coefs[slot + comp] = coefficient
            rows.append((coefs, -fixed, 1.0))
    return rows


def _cross_extent(surface, edge: str, t: float) -> float:
    """How far the patch reaches across itself at one boundary station."""
    if edge in ("south", "north"):
        return abs(surface.point_at(t, 1.0) - surface.point_at(t, 0.0))
    return abs(surface.point_at(1.0, t) - surface.point_at(0.0, t))


def _support_reach(support, spine, t: float) -> float:
    """How far the support extends from its spine, along the support itself.

    The spine is located on the support by projection; the reach is the
    shorter of the two distances to the support's edges in that direction.
    For a symmetric ribbon of width ``w`` that is ``w / 2``, for a one-sided
    ribbon ``w`` -- which is exactly the span a tangency strength is relative
    to.
    """
    point = spine.point_at(t)
    _projected, (u, v_spine) = support.project(point)
    distances = []
    for v_edge in (0.0, 1.0):
        try:
            edge_point = support.point_at(u, v_edge)
        except Exception:  # pragma: no cover - defensive, parameterisation dependent
            continue
        distances.append(abs(edge_point - point))
    distances = [d for d in distances if d > _EPS]
    if not distances:
        raise G1Error("could not measure the support's extent from the spine")
    return min(distances)


def support_curvature_field(support, spine, towards=None, step: float = 1e-5, tol: float = 1e-4):
    """Normal-section curvature of *support* across its spine.

    ``kappa = (d2S/dcross2 . N) / |dS/dcross|^2``, invariant under
    reparameterisation of the cross direction, so it is directly comparable
    with the patch's own number. The second derivative comes from a finite
    difference of the analytic first derivative.
    """
    at = support_fields(support, spine, towards=towards, tol=tol)

    def curvature(t: float) -> float:
        point = spine.point_at(t)
        _projected, (u, v) = support.project(point)
        tangent = spine.tangent_at(t)
        length = (tangent.x**2 + tangent.y**2 + tangent.z**2) ** 0.5
        t_hat = (tangent.x / length, tangent.y / length, tangent.z / length)
        su, sv = support.tangents_at(u, v)
        candidates = [(0, su), (1, sv)]
        best = None
        for which, candidate in candidates:
            cl = (candidate.x**2 + candidate.y**2 + candidate.z**2) ** 0.5
            if cl < 1e-12:
                continue
            align = abs(
                candidate.x * t_hat[0] + candidate.y * t_hat[1] + candidate.z * t_hat[2]
            ) / cl
            if best is None or align < best[0]:
                best = (align, which)
        if best is None:
            raise G1Error("the support has no transverse derivative at the spine")
        which = best[1]
        normal, _slope = at(t)
        if which == 0:
            first = support.tangents_at(u, v)[0]
            later = support.tangents_at(u + step, v)[0]
        else:
            first = support.tangents_at(u, v)[1]
            later = support.tangents_at(u, v + step)[1]
        d = (first.x, first.y, first.z)
        dd = (
            (later.x - first.x) / step,
            (later.y - first.y) / step,
            (later.z - first.z) / step,
        )
        length_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        if length_sq < 1e-12:
            raise G1Error("the support's cross derivative vanishes at the spine")
        return (
            dd[0] * normal[0] + dd[1] * normal[1] + dd[2] * normal[2]
        ) / length_sq

    return curvature


def support_constraint(support, spine, towards=None, tol: float = 1e-4) -> dict:
    """A G1 field spec for one edge, carrying its support and spine.

    Use this instead of a bare normal callable when a tangency strength is
    wanted: the solver needs the support's own slope and reach to scale it.
    """
    at = support_fields(support, spine, towards=towards, tol=tol)
    return {
        "normal": lambda t: at(t)[0],
        "slope": lambda t: at(t)[1],
        "support": support,
        "spine": spine,
    }


def solve_g1_constraints(
    surface,
    params_u: Sequence[float],
    params_v: Sequence[float],
    edge_normal_fields: dict,
    ridge: float = 1e-10,
    corner_stations: int = 1,
    tangency: float | None = None,
    g2_edges: "Sequence[str] | None" = None,
):
    """P-1: enforce the G1 tangent-plane rows on a P-0 Coons patch.

    Args:
        surface:            the P-0 discrete Coons surface (not mutated).
        params_u/params_v:  the station parameters the patch was built on.
        edge_normal_fields: ``{"south": field, ...}`` with any of
                            `EDGE_NAMES`; an absent edge stays G0.
        ridge:              relative Tikhonov weight -- ties the under-
                            determined system to the smallest patch change.
        g2_edges:           edges whose *curvature* across the boundary must
                            also match the support's (G2). Runs as a second
                            pass over the G1 result, so the tangent plane is
                            already in place. Needs the support and spine in
                            the field spec.
        tangency:           G1 only fixes the *direction* of the boundary
                            cross-derivative, leaving its magnitude to the
                            solve. Give a factor to control that magnitude:
                            ``1.0`` keeps the Coons base's in-plane slope,
                            larger values leave the edge more stiffly (the
                            surface pulls away from the tangent plane faster),
                            smaller values more softly. ``None`` leaves the
                            minimum-norm magnitude.

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
    movable = set()
    for i in range(1, n_u - 1):
        movable.add((i, 1))
        movable.add((i, n_v - 2))
    for j in range(1, n_v - 1):
        movable.add((1, j))
        movable.add((n_u - 2, j))
    band = sorted(movable)
    index = {point: 3 * k for k, point in enumerate(band)}
    if not band:
        raise G1Error("patch net too small to carry G1 constraints")

    rows = []

    def dot3(a, b) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def add_edge_rows(samples, bases, factor, inner_key, outer_key, edge, spec) -> None:
        for t, normal in samples:
            # the Coons base's own cross-derivative at this station, and the
            # control points that carry it
            current = [0.0, 0.0, 0.0]
            terms = []
            for slot, weight in bases(t):
                i_slot, j_slot = slot[0], slot[1]
                inner = net[i_slot][j_slot]
                outer_i, outer_j = outer_key(i_slot, j_slot)
                outer = net[outer_i][outer_j]
                coefficient = factor * weight
                for comp in range(3):
                    current[comp] += coefficient * (inner[comp] - outer[comp])
                terms.append((index.get(inner_key(i_slot, j_slot)), coefficient))

            if tangency is None:
                # direction only: the off-plane component must vanish, the
                # magnitude stays whatever the minimum-norm solve leaves
                coefs = {}
                for base_index, coefficient in terms:
                    if base_index is None:
                        continue
                    for comp in range(3):
                        coefs[base_index + comp] = coefficient * normal[comp]
                rows.append((coefs, -dot3(current, normal)))
                continue

            # strength given: aim the cross-derivative at the support's own
            # slope, scaled so the patch leaves the edge `tangency` times as
            # fast as the support does. One row per component.
            slope = _slope_of(spec, t)
            if slope is None:
                raise G1Error(
                    "a tangency strength needs the support's slope: build the "
                    "fields with patch_g1.support_fields"
                )
            slope_length = (
                slope[0] ** 2 + slope[1] ** 2 + slope[2] ** 2
            ) ** 0.5
            if slope_length < _EPS:
                raise G1Error("the support slope is degenerate at a station")
            # support slope per unit distance, times the patch's cross extent
            unit = [slope[comp] / slope_length for comp in range(3)]
            support = spec.get("support") if isinstance(spec, dict) else None
            spine = spec.get("spine") if isinstance(spec, dict) else None
            if support is None or spine is None:
                raise G1Error(
                    "a tangency strength needs the support and its spine in the "
                    "field spec -- build it with patch_g1.support_constraint"
                )
            reach = _cross_extent(surface, edge, t)
            support_reach = _support_reach(support, spine, t)
            magnitude = tangency * (slope_length / support_reach) * reach
            target = [magnitude * unit[comp] for comp in range(3)]
            for comp in range(3):
                coefs = {}
                for base_index, coefficient in terms:
                    if base_index is None:
                        continue
                    coefs[base_index + comp] = coefficient
                rows.append((coefs, target[comp] - current[comp]))

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
                edge,
                field,
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
                edge,
                field,
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
    result = Surface(
        control_points=out,
        uknots=list(surface.uknots),
        vknots=list(surface.vknots),
        umults=list(surface.umults),
        vmults=list(surface.vmults),
        udegree=qu,
        vdegree=qv,
    )
    if not g2_edges:
        return result
    return _refine_g2(
        result,
        params_u,
        params_v,
        edge_normal_fields,
        g2_edges,
        ridge=ridge,
        corner_stations=corner_stations,
        tangency=tangency,
    )


def support_fields(support, spine, towards=None, tol: float = 1e-4):
    """The (normal, slope) fields of a support surface along its spine.

    The normal field is the support's unit normal at the spine -- the G1
    tangent plane. The slope field is the support's own cross-derivative
    there, normal component removed and oriented towards *towards*: the rate
    at which the support itself leaves the tangent plane. That is what a
    tangency strength is measured against.

    Args:
        support: any `Surface` carrying the constraint plane.
        spine:   the `Curve` the constraint is attached to.
        towards: a point inside the patch, used to orient the slope.
        tol:     how far the spine may sit off the support before raising.

    Returns:
        ``(normal_field, slope_field)``, each a callable of the spine
        parameter. Compose with `patch_np.edge_curve_params` for patch
        parameter space, or use `support_constraint` for the whole spec.
    """

    def at(t: float):
        point = spine.point_at(t)
        projected, (u, v) = support.project(point)
        gap = abs(projected - point)
        if gap > tol:
            raise G1Error(
                f"the spine misses its support by {gap:.3e} at t={t:.4f} "
                f"(tol {tol:.1e}) -- wrong support for this curve?"
            )
        normal = support.normal_at(u, v)
        n = _unit((normal.x, normal.y, normal.z))
        tangent = spine.tangent_at(t)
        length = (tangent.x**2 + tangent.y**2 + tangent.z**2) ** 0.5
        t_hat = (tangent.x / length, tangent.y / length, tangent.z / length)

        # the transverse derivative: the one least aligned with the spine
        best, best_align = None, 2.0
        for candidate in support.tangents_at(u, v):
            cl = (candidate.x**2 + candidate.y**2 + candidate.z**2) ** 0.5
            if cl < 1e-12:
                continue
            align = abs(
                candidate.x * t_hat[0] + candidate.y * t_hat[1] + candidate.z * t_hat[2]
            ) / cl
            if align < best_align:
                best, best_align = candidate, align
        if best is None:
            raise G1Error("the support has no transverse derivative at the spine")

        # strip the normal component, orient towards the patch
        along = best.x * n[0] + best.y * n[1] + best.z * n[2]
        slope = (best.x - along * n[0], best.y - along * n[1], best.z - along * n[2])
        if towards is not None:
            inward = (towards.x - point.x, towards.y - point.y, towards.z - point.z)
            if slope[0] * inward[0] + slope[1] * inward[1] + slope[2] * inward[2] < 0.0:
                slope = (-slope[0], -slope[1], -slope[2])
        return n, slope

    return at


def normal_field_from_support(support, spine, tol: float = 1e-4):
    """The normal field alone -- a shorthand for `support_fields`."""
    at = support_fields(support, spine, tol=tol)

    def normal_at(t: float) -> tuple:
        return at(t)[0]

    return normal_at


def _refine_g2(
    surface,
    params_u,
    params_v,
    edge_normal_fields,
    g2_edges,
    ridge: float = 1e-10,
    corner_stations: int = 1,
    tangency: float | None = None,
):
    """Second pass: match the support's cross-curvature on *g2_edges*.

    The G1 rows are rebuilt unchanged (so the tangent plane cannot drift) and
    the curvature rows are added:

        (S_vv(u,0) . N) = kappa_support(u) * |S_v(u,0)|^2

    with ``|S_v|`` and ``kappa_support`` taken from the G1 result. The band is
    widened to the poles that carry ``S_vv`` (rows 1 and 2 at each end, and the
    matching columns).
    """
    from geometrykit.geometry.surface import Surface

    qu, qv = surface.udegree, surface.vdegree
    flat_u = _flat_knots(surface.uknots, surface.umults)
    flat_v = _flat_knots(surface.vknots, surface.vmults)
    span_v_lo = flat_v[qv + 1] - flat_v[qv]
    span_v_hi = flat_v[-qv - 1] - flat_v[-qv - 2]
    span_u_lo = flat_u[qu + 1] - flat_u[qu]
    span_u_hi = flat_u[-qu - 1] - flat_u[-qu - 2]
    low_v, high_v = _second_deriv_weights(flat_v, qv)
    low_u, high_u = _second_deriv_weights(flat_u, qu)

    net = [[(p.x, p.y, p.z) for p in row] for row in surface.control_points]
    n_u = len(net)
    n_v = len(net[0])

    movable = set()
    for i in range(1, n_u - 1):
        for j in (1, 2, n_v - 3, n_v - 2):
            movable.add((i, j))
    for j in range(1, n_v - 1):
        for i in (1, 2, n_u - 3, n_u - 2):
            movable.add((i, j))
    band = sorted(movable)
    index = {point: 3 * k for k, point in enumerate(band)}
    if not band:
        raise G1Error("patch net too small to carry G2 constraints")

    rows = []

    def dot3(a, b) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    # ── the G1 rows again, so the tangent plane is preserved exactly
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
            for t, normal in samples:
                coefs = {}
                residual = 0.0
                for i, n_i in _nonzero_basis(t, qu, flat_u).items():
                    if not 0 <= i < n_u:
                        continue
                    inner = net[i][inner_row]
                    outer = net[i][outer_row]
                    delta = (
                        inner[0] - outer[0],
                        inner[1] - outer[1],
                        inner[2] - outer[2],
                    )
                    coefficient = factor * n_i
                    residual += coefficient * dot3(delta, normal)
                    base_index = index.get((i, inner_row))
                    if base_index is None:
                        continue
                    for comp in range(3):
                        coefs[base_index + comp] = coefficient * normal[comp]
                rows.append((coefs, -residual))
        else:
            samples = _field_samples(field, params_v, corner_stations)
            low = edge == "west"
            factor = (qu / span_u_lo) if low else (qu / span_u_hi)
            inner_col = 1 if low else n_u - 2
            outer_col = 0 if low else n_u - 1
            for t, normal in samples:
                coefs = {}
                residual = 0.0
                for j, n_j in _nonzero_basis(t, qv, flat_v).items():
                    if not 0 <= j < n_v:
                        continue
                    inner = net[inner_col][j]
                    outer = net[outer_col][j]
                    delta = (
                        inner[0] - outer[0],
                        inner[1] - outer[1],
                        inner[2] - outer[2],
                    )
                    coefficient = factor * n_j
                    residual += coefficient * dot3(delta, normal)
                    base_index = index.get((inner_col, j))
                    if base_index is None:
                        continue
                    for comp in range(3):
                        coefs[base_index + comp] = coefficient * normal[comp]
                rows.append((coefs, -residual))

    # ── the curvature rows
    for edge in EDGE_NAMES:
        if edge not in g2_edges:
            continue
        field = edge_normal_fields.get(edge)
        if field is None:
            continue
        support = field.get("support") if isinstance(field, dict) else None
        spine = field.get("spine") if isinstance(field, dict) else None
        if support is None or spine is None:
            raise G1Error(
                f"a G2 constraint on '{edge}' needs the support and its spine in "
                "the field spec -- build it with patch_g1.support_constraint"
            )
        towards = Vec(
            sum(net[i][j][0] for i in (1, n_u - 2) for j in (1, n_v - 2)) / 4.0,
            sum(net[i][j][1] for i in (1, n_u - 2) for j in (1, n_v - 2)) / 4.0,
            sum(net[i][j][2] for i in (1, n_u - 2) for j in (1, n_v - 2)) / 4.0,
        )
        curvature = support_curvature_field(support, spine, towards=towards)
        low = edge in ("south", "west")
        weights = low_v if edge in ("south", "north") else low_u
        if not low:
            weights = high_v if edge in ("south", "north") else high_u
        varying = params_u if edge in ("south", "north") else params_v
        flat_varying = flat_u if edge in ("south", "north") else flat_v
        degree_varying = qu if edge in ("south", "north") else qv
        n_varying = n_u if edge in ("south", "north") else n_v
        samples = _field_samples(field, varying, corner_stations)

        for t, normal in samples:
            # the support's curvature and the patch's own first derivative
            patch_params = (
                params_u if edge in ("south", "north") else params_v
            )
            del patch_params
            t_support = t
            kappa = curvature(t_support)
            if edge == "south":
                first = surface.tangents_at(t, 0.0)[1]
            elif edge == "north":
                first = surface.tangents_at(t, 1.0)[1]
            elif edge == "west":
                first = surface.tangents_at(0.0, t)[0]
            else:
                first = surface.tangents_at(1.0, t)[0]
            speed_sq = first.x**2 + first.y**2 + first.z**2
            target = kappa * speed_sq

            coefs = {}
            residual = 0.0
            for i, n_i in _nonzero_basis(t, degree_varying, flat_varying).items():
                if not 0 <= i < n_varying:
                    continue
                for pole_index, weight in weights.items():
                    if not 0 <= pole_index < (n_v if edge in ("south", "north") else n_u):
                        continue
                    if edge in ("south", "north"):
                        point = (i, pole_index)
                        value = net[i][pole_index]
                    else:
                        point = (pole_index, i)
                        value = net[pole_index][i]
                    coefficient = n_i * weight
                    residual += coefficient * dot3(value, normal)
                    base_index = index.get(point)
                    if base_index is None:
                        continue
                    for comp in range(3):
                        coefs[base_index + comp] = coefficient * normal[comp]
            rows.append((coefs, target - residual))

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


def constrained_coons_patch(
    curves: Sequence,
    edge_normal_fields: dict,
    nu: int = 12,
    nv: int = 12,
    tol: float = 1e-6,
    degree: int = 3,
    ridge: float = 1e-10,
    corner_stations: int = 1,
    tangency: float | None = None,
    g2: bool = False,
    fill: bool = False,
    parametrization: str = "chord",
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
        tangency:           optional tangency strength -- the boundary
                            cross-derivative's magnitude as a multiple of the
                            support's own slope (``1.0`` matches it, higher is
                            stiffer). ``None`` leaves it to the minimum-norm
                            solve.
        fill:               base the patch on the energy-minimising plate
                            fill instead of the Coons blend. The boundary rows
                            are identical; the interior is the thin-plate
                            solution through that boundary, so the continuity
                            solve starts from a smooth surface.
        parametrization:    how the patch stations are spread -- ``"chord"``
                            (the curve's own parameterisation) or
                            ``"arclength"`` (even in space, uniform knots).
                            See `patch_np.coons_parametrization`.
        g2:                 also match the support's cross-curvature across
                            every G1 edge (curvature continuity), as a second
                            pass over the G1 result. Needs the support and
                            spine in each field spec.
        corner_stations:    stations dropped at each end of an edge (the
                            corner band, where the boundary curves pin
                            the cross-derivative).

    Returns:
        ``(surface, corners, stations)`` -- the G1-adjusted surface, the
        corner sheet, and ``(params_u, params_v)`` so callers can probe the
        patch at the very stations the constraints were placed on.
    """
    surface, corners = discrete_coons_patch(
        curves, nu=nu, nv=nv, tol=tol, degree=degree, parametrization=parametrization
    )
    params_u, params_v, _oriented = coons_parametrization(
        curves, nu=nu, nv=nv, tol=tol, parametrization=parametrization
    )
    if fill:
        # The base is the energy-minimising plate fill, not the Coons blend:
        # the boundary rows are identical (same stations, same boundary curves)
        # but the interior is the thin-plate solution, so the continuity solve
        # below starts from a surface that already wants to be smooth instead
        # of from a flat blend with the bending at the edges.
        surface, corners = discrete_plate_fill(
            curves,
            nu=nu,
            nv=nv,
            tol=tol,
            degree=degree,
            parametrization=parametrization,
        )

    fixed = solve_g1_constraints(
        surface,
        params_u,
        params_v,
        edge_normal_fields,
        ridge=ridge,
        corner_stations=corner_stations,
        tangency=tangency,
        g2_edges=list(EDGE_NAMES) if g2 else None,
    )
    return fixed, corners, (params_u, params_v)