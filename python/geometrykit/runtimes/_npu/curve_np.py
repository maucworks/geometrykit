# This file was generated with the assistance of an AI coding tool.
"""Pure-Python curve kernels (runtime `_npu`): global interpolation + assembly.

M3 from the OCC-exit plan:

- ``interpolate`` — global interpolation (NURBS Book A9.1); with G1
  end-tangent conditions the system enlarges A9.2-style: control point
  count = data count + tangents supplied, two derivative-condition rows,
  averaged knots over the data parameters.
- ``_gauss_solve`` — dense Gaussian elimination with vector rhs (banded,
  diagonally dominant systems; interactive sizes only).
- ``join_two`` — C0 knot-vector assembly of two equal-degree
  non-rational curves: control points concatenated at the shared seam,
  B's clamped head (degree+1 flats) absorbed into A's clamped tail so
  the seam multiplicity is exactly `degree` (C0 semantics, matching
  ``GeomConvert_CompCurveToBSplineCurve``).
- ``group_and_orient`` — the 4-way endpoint-permutation orientation used
  by ``Curve.assemble`` (ported verbatim from the OCC path).

Rational input on interpolation is out of scope (ifckit's callers stay
non-rational); joining rational to non-rational raises instead of
silently dropping weights.
"""

from __future__ import annotations

from typing import Optional, Sequence

_EPS = 1e-12


class Vec3:
    """Tiny xyz record for kernel returns."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


def chord_params(points: Sequence) -> list:
    """Cumulative chord-length parameters in ``[0, 1]`` (first 0, last 1)."""
    total = 0.0
    acc = [0.0]
    for i in range(1, len(points)):
        total += abs(points[i] - points[i - 1])
        acc.append(total)
    if total < _EPS:
        return [0.0] * len(points)
    return [a / total for a in acc]


def averaged_inner_knots(
    params: Sequence[float], q: int, count: int, window_offset: int = 0
) -> list:
    """Averaged inner knots.

    Window ``j`` (1-based): mean of ``params[j + window_offset ..
    j + q − 1 + window_offset]`` (0-based parameters). Note the window starts
    at ``params[j + window_offset]``, not at ``params[j − 1 + window_offset]``:
    with ``window_offset = 0`` it therefore covers the *last* data parameters
    (Piegl's eq. 9.8 window shifted one step forward). That placement is what
    the standalone curve interpolation matches OCCT with, and what the patch
    grid stays solvable with; ``window_offset = −1`` is Piegl's literal rule
    and leaves the final span wide. The A9.2 tangent-case layout uses
    ``window_offset = −1`` on a data list enlarged by two. Windows that run
    past the data clamp to the last parameter (A9.1 tail behaviour).
    """
    out = []
    for j in range(1, count + 1):
        s = 0.0
        used = 0
        for a in range(j + window_offset, j + window_offset + q):
            if 0 <= a < len(params):
                s += params[a]
                used += 1
        out.append(s / q if used else params[-1])
    return out


def _nonzero_basis(u: float, p: int, uflat: Sequence[float]) -> dict:
    """Non-zero degree-p basis values at *u* → ``{id: N}`` (A2.2).

    Edge-aware span: the exact right limit evaluates the last live span
    (Piegl's `u == U[n+1] → n` special case).
    """
    top = len(uflat) - p - 2  # Piegl n
    if u >= uflat[top + 1]:
        span = top
    elif u <= uflat[p]:
        span = p
    else:
        lo, hi = p, top + 1
        mid = (lo + hi) // 2
        while u < uflat[mid] or u >= uflat[mid + 1]:
            if u < uflat[mid]:
                hi = mid
            else:
                lo = mid
            mid = (lo + hi) // 2
        span = mid
    left = [0.0] * (p + 1)
    right = [0.0] * (p + 1)
    N = [0.0] * (p + 1)
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - uflat[span + 1 - j]
        right[j] = uflat[span + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            t = N[r] / denom if abs(denom) > _EPS else 0.0
            N[r] = saved + right[r + 1] * t
            saved = left[j - r] * t
        N[j] = saved
    return {span - p + k: N[k] for k in range(p + 1)}


def _low_row(u: float, p: int, uflat: Sequence[float]) -> dict:
    """Degree-(p−1) non-zero row at *u*, with clamp-continuation at both edges."""
    lo = uflat[p] if p < len(uflat) else 0.0
    hi = uflat[len(uflat) - p - 1] if len(uflat) - p - 1 >= 0 else 0.0
    eps = (hi - lo) * 1e-9 if hi > lo else 0.0

    if eps > 0.0:
        if u >= hi:
            return _nonzero_basis(hi - eps, p - 1, uflat)
        if u <= lo and p > 0:
            return _nonzero_basis(lo + eps, p - 1, uflat)
    return _nonzero_basis(u, p - 1, uflat)


def _basis_and_deriv(u: float, p: int, uflat: Sequence[float]) -> tuple:
    """``({id: N}, {id: dN/du})`` at *u*.

    The VALUE row evaluates at the exact parameter; the derivative row uses
    the one-sided limit (inside the live span) at clamped edges — OCC's `D1`
    endpoint convention: derivative by continuation, value exact.
    """
    Nu = _nonzero_basis(u, p, uflat)
    if p == 0:
        return Nu, {i: 0.0 for i in Nu}
    left_limit = uflat[p]
    right_limit = uflat[len(uflat) - p - 1]

    def _limit_point(x: float) -> float:
        eps = (right_limit - left_limit) * 1e-9 if right_limit > left_limit else 0.0
        if eps > 0.0:
            if x >= right_limit:
                return right_limit - eps
            if x <= left_limit:
                return left_limit + eps
        return x

    low = _nonzero_basis(_limit_point(u), p - 1, uflat)
    top = len(uflat) - 1
    ders: dict = {}
    for i in Nu:
        a = uflat[i + p] - uflat[i] if i + p <= top else 0.0
        b = uflat[i + p + 1] - uflat[i + 1] if i + p + 1 <= top else 0.0
        t1 = (p * low.get(i, 0.0) / a) if abs(a) > _EPS else 0.0
        t2 = (p * low.get(i + 1, 0.0) / b) if abs(b) > _EPS else 0.0
        ders[i] = t1 - t2
    return Nu, ders


def _flat_unique(flat: Sequence[float]) -> tuple:
    knots: list = []
    mults: list = []
    for val in flat:
        if knots and abs(val - knots[-1]) < 1e-9:
            mults[-1] += 1
        else:
            knots.append(val)
            mults.append(1)
    return knots, mults


def _gauss_solve(rows: list, unknowns: int, dims: int = 3) -> list:
    """Dense Gaussian elimination with vector rhs.

    rows = ``({col: coef}, (rhs_x, rhs_y, rhs_z))`` pairs; returns the
    per-unknown solution as ``[(x, y, z), …]``.
    """
    A = [[0.0] * (unknowns + dims) for _ in range(len(rows))]
    for r, (coefs, rhs) in enumerate(rows):
        for i, c in coefs.items():
            if 0 <= i < unknowns:
                A[r][i] = float(c)
        for d in range(dims):
            A[r][unknowns + d] = float(rhs[d])
    n = len(rows)
    for col in range(unknowns):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < _EPS:
            raise RuntimeError("interpolation system is singular (degenerate data/tangents?)")
        A[col], A[piv] = A[piv], A[col]
        for r in range(n):
            if r == col or abs(A[r][col]) < _EPS:
                continue
            f = A[r][col] / A[col][col]
            for c in range(col, unknowns + dims):
                A[r][c] -= f * A[col][c]
    # full Jordan sweep → each row keeps one unknown at column col
    sol: list = [None] * unknowns
    for col in range(unknowns):
        row = next((r for r in range(n) if abs(A[r][col]) > _EPS), None)
        if row is None:
            raise RuntimeError("interpolation system lost a pivot row")
        vals = tuple(
            A[row][unknowns + d_] / A[row][col] for d_ in range(dims)
        )
        sol[col] = vals
    return sol


def interpolate(
    points: Sequence,
    degree: int = 3,
    tan_start: Optional = None,
    tan_end: Optional = None,
    knots_param: Optional[Sequence[float]] = None,
):
    """Global interpolation of *points* (NURBS Book A9.1 / A9.2 end conditions).

    Args:
        points:    Data points (≥ 2); the curve passes through all of them.
        degree:    Requested degree — clamped to ``len(points) − 1``.
        tan_start/tan_end: Optional tangent directions (G1). The end
            derivative magnitude is the natural chord one
            ``(params₁ − params₀)/degree`` (unscaled convention).
        knots_param: Optional explicit parameters in ``[0, 1]``
            (length must equal points); chord params otherwise.

    Returns:
        ``(control_points, knots, multiplicities)`` — control_points as
        ``(x, y, z)`` tuples.
    """
    r = len(points)
    if r < 2:
        raise ValueError("Need at least 2 points")
    with_t_start = tan_start is not None
    with_t_end = tan_end is not None
    extra = (1 if with_t_start else 0) + (1 if with_t_end else 0)
    unknowns = r + extra
    q = max(1, min(degree, unknowns - 1))
    params = list(knots_param if knots_param is not None else chord_params(points))
    if len(params) != r:
        raise ValueError("params length must match points")

    inner = unknowns - q - 1
    flat: list = (
        [0.0] * (q + 1)
        + averaged_inner_knots(params, q, inner, window_offset=-1 if extra else 0)
        + [1.0] * (q + 1)
    )
    if len(flat) != unknowns + q + 1:
        raise RuntimeError(
            f"interpolation knot layout broken: {len(flat)} vs {unknowns + q + 1}"
        )

    rows: list = []
    for k in range(r):
        Nu = _nonzero_basis(params[k], q, flat)
        rows.append((Nu, (points[k].x, points[k].y, points[k].z)))

    if with_t_start and r >= 2:
        den = params[1] - params[0]
        sc = q / den
        rows.insert(0, ({0: -sc, 1: sc}, (tan_start.x, tan_start.y, tan_start.z)))
    if with_t_end and r >= 2:
        den = params[r - 1] - params[r - 2]
        sc = q / den
        last = unknowns - 2
        rows.append(({last: -sc, last + 1: sc}, (tan_end.x, tan_end.y, tan_end.z)))

    if len(rows) != unknowns:
        raise RuntimeError(f"rows {len(rows)} vs unknowns {unknowns}: system shape mismatch")
    sol = _gauss_solve(rows, unknowns)
    knots, mults = _flat_unique(flat)
    cpts = [(sol[i][0], sol[i][1], sol[i][2]) for i in range(unknowns)]
    return cpts, knots, mults


def join_two(a, b, tol: float = 1e-6):
    """C0-join two equal-degree non-rational curves at the shared seam.

    Returns ``(control_points, knots, multiplicities)`` or ``None`` when
    the seam misses *tol* or the degrees differ (caller breaks the group).
    """
    if a.rational or b.rational:
        raise ValueError("joining rational curves is not supported by the C0 join")
    if a.degree != b.degree:
        return None
    if abs(a.point_at(1.0) - b.point_at(0.0)) > tol:
        return None
    cpts = list(a.points) + list(b.points[1:])
    flat = a._uknots[:-1] + [
        f + (a._uknots[-1] - b._uknots[0]) for f in b._uknots[a.degree + 1 :]
    ]
    knots, mults = _flat_unique(flat)
    return cpts, knots, mults


def group_and_orient(curves: Sequence, tol: float = 1e-6) -> list:
    """G0 grouping + 4-way endpoint-permutation orientation.

    Ported from ifckit's ``Curve.assemble`` (OCC path) — identical
    grouping semantics: curves not G0-connected within *tol* start a new
    group; orientation is repaired by reversal when needed.

    Returns ``list[list[Curve]]`` (oriented groups).
    """
    if not curves:
        return []
    oriented = [curves[0]]
    groups: list[list] = []

    def _p(i):
        return oriented[-1].point_at(0), oriented[-1].point_at(1)

    def _dist(a, b) -> float:
        return abs(a - b)

    for i in range(1, len(curves)):
        prev_start, prev_end = _p(-1)
        curr_start, curr_end = curves[i].point_at(0), curves[i].point_at(1)

        fwd = _dist(prev_end, curr_start)
        rev = _dist(prev_end, curr_end)
        p_rev = _dist(prev_start, curr_start)
        p_rev_r = _dist(prev_start, curr_end)

        if min(fwd, rev, p_rev, p_rev_r) > tol:
            groups.append(oriented)
            oriented = [curves[i]]
            continue

        if fwd <= tol:
            oriented.append(curves[i])
        elif rev <= tol:
            oriented.append(curves[i].reverse())
        elif p_rev <= tol:
            oriented[-1] = oriented[-1].reverse()
            oriented.append(curves[i])
        else:
            oriented[-1] = oriented[-1].reverse()
            oriented.append(curves[i].reverse())

    groups.append(oriented)
    return groups


def arclength_divide(
    point_at,
    total: float,
    num: "int | None" = None,
    dist: "float | None" = None,
    even: bool = True,
) -> list:
    """Kernel arc-length division: normalized parameters ``∈ [0, 1]``.

    A dense chord table over ``point_at`` is inverted per target
    distance (binary search + linear interpolation). Arc-length
    uniformity holds within a few percent on fair curves (the old
    parameter-uniform fallback measured 2.7x gap spread on a Hermite
    wiggle); straight chords between outputs vary more where the
    curve bends, which is geometric, not error.

    Args:
        point_at: callable ``t → xyz`` (t in [0, 1]).
        total:    curve length in the same units as ``dist``.
        num:      point count (endpoints included).
        dist:     target spacing; ``even`` distributes evenly with the
                  endpoints included, otherwise fixed steps from the
                  start plus the exact endpoint.
        even:     spacing rule for *dist* (see above).

    Returns:
        Ascending parameter list (first 0.0, last 1.0).
    """
    import math

    if total < _EPS:
        raise ValueError("Cannot divide a zero-length curve")
    table_n = max(64, 8 * (num if num is not None else 16))
    samples = [point_at(i / table_n) for i in range(table_n + 1)]
    cumulative = [0.0]
    for a, b in zip(samples, samples[1:]):
        cumulative.append(
            cumulative[-1]
            + math.sqrt(
                (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2
            )
        )
    table_total = cumulative[-1]
    if table_total < _EPS:
        raise ValueError("Cannot divide a zero-length curve")
    if num is not None:
        targets = [table_total * k / (num - 1) for k in range(num)]
    elif even:
        segs = max(1, math.ceil(table_total / dist))
        targets = [table_total * k / segs for k in range(segs + 1)]
    else:
        targets = []
        d = 0.0
        while d < table_total:
            targets.append(d)
            d += dist
        targets.append(table_total)

    out = []
    for target in targets:
        lo, hi = 0, table_n
        while lo < hi:
            mid = (lo + hi) // 2
            if cumulative[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        if lo <= 0:
            out.append(0.0)
        elif lo > table_n:
            out.append(1.0)
        else:
            span = cumulative[lo] - cumulative[lo - 1]
            if span <= 0.0:
                out.append(lo / table_n)
            else:
                out.append((lo - 1 + (target - cumulative[lo - 1]) / span) / table_n)
    return out
