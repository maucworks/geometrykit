# This file was generated with the assistance of an AI coding tool.
"""Pure-Python tensor-product NURBS surface kernels (runtime `_npu`).

The M2/M4-early evaluation family from the OCC-exit plan:

- ``eval_point`` — tensor de Boor with homogeneous divide (rational-safe);
- ``eval_tangents`` — analytic first-order derivatives (eq. 3.8 + quotient rule);
- ``eval_normal`` — unit cross product of the analytic tangents;
- ``project_point`` — multi-start Gauss–Newton in (u, v), domain-clamped.

Derivative correctness note: the de Boor recursion here captures the
intermediate degree rows, so derivatives use the difference formula

    d/du N_i^p = p·N_i^{p−1}/(U_{i+p} − U_i) − p·N_{i+1}^{p−1}/(U_{i+p+1} − U_{i+1})

with degenerate (repeated-clamp) denominators contributing exactly zero —
correct one-sided values at clamped domain edges. It deliberately does not
reuse ``curve._cox_de_boor_deriv``, whose clamped-edge convention the 1D
kernel did not need yet. A Rust port must port THIS file, not its imports.

Knot layout: surfaces store separate knot vectors plus multiplicity lists
(``uknots``/``umults``); kernels use the expanded vector — the active domain
is ``uflat[p] .. uflat[-p-1]``.
"""

from __future__ import annotations

import math
from typing import Sequence


class Vec3:
    """Minimal xyz triple produced by kernels (duck-typed as the package's ``Vec``)."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


def flat_knots(knots: Sequence[float], mults: Sequence[int]) -> list:
    """Expanded knot vector from a knot/multiplicity pair."""
    out: list[float] = []
    for k, m in zip(knots, mults):
        out += [k] * m
    return out


def find_span(u: float, p: int, uflat: Sequence[float]) -> int:
    """Span index containing *u* (Piegl A2.1).

    Special case follows Piegl: ``u == U[n+1]`` returns ``n`` (the top
    control index, ``len(uflat) - p - 2``), so the exact right limit
    evaluates the last live basis set.
    """
    n = len(uflat) - p - 2
    if u >= uflat[n + 1]:
        return n
    if u <= uflat[p]:
        return p
    lo, hi = p, n + 1
    mid = (lo + hi) // 2
    while u < uflat[mid] or u >= uflat[mid + 1]:
        if u < uflat[mid]:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) // 2
    return mid


def _nonzero_row(u: float, p: int, uflat: Sequence[float]) -> dict:
    """Non-zero degree-p basis values at *u*: ``{id: N_id^p(u)}`` (A2.2)."""
    span = find_span(u, p, uflat)
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
            t = N[r] / denom if abs(denom) > 1e-12 else 0.0
            N[r] = saved + right[r + 1] * t
            saved = left[j - r] * t
        N[j] = saved
    return {span - p + k: N[k] for k in range(p + 1)}


def _lower_row(u: float, p: int, uflat: Sequence[float], Nu: dict) -> dict:
    """Legacy helper kept name-compatible; superseded by `_basis_and_deriv`'s
    right-limit-aware low row."""
    return _nonzero_row(u, p - 1, uflat) if p > 0 else {find_span(u, 0, uflat): 1.0}


def _basis_and_deriv(u: float, p: int, uflat: Sequence[float]) -> tuple:
    """``({id: N_id^p}, {id: dN_id^p/du})`` at *u* (eq. 3.8 derivative).

    Right-limit handling: at the exact domain end the derivative rows are
    evaluated one representable step inside the last live span (left
    continuation — the slope the surface actually has there), matching
    OCC's endpoint `D1` convention.
    """
    Nu = _nonzero_row(u, p, uflat)  # value row at the exact parameter
    if p == 0:
        return Nu, {i: 0.0 for i in Nu}
    left_limit = uflat[p]
    right_limit = uflat[len(uflat) - p - 1]
    eps = (right_limit - left_limit) * 1e-9 if right_limit > left_limit else 0.0

    def _limit_point(x: float) -> float:
        if eps > 0.0:
            if x >= right_limit:
                return right_limit - eps
            if x <= left_limit:
                return left_limit + eps
        return x

    # derivative row at the one-sided limit (value row stays exact)
    low = _nonzero_row(_limit_point(u), p - 1, uflat)
    top = len(uflat) - 1
    ders: dict = {}
    for i in Nu:
        a = uflat[i + p] - uflat[i] if i + p <= top else 0.0
        b = uflat[i + p + 1] - uflat[i + 1] if i + p + 1 <= top else 0.0
        t1 = (p * low.get(i, 0.0) / a) if abs(a) > 1e-12 else 0.0
        t2 = (p * low.get(i + 1, 0.0) / b) if abs(b) > 1e-12 else 0.0
        ders[i] = t1 - t2
    return Nu, ders


class _Data:
    """Prepared surface payload: expanded knots + grid weights (or None)."""

    __slots__ = ("uflat", "vflat", "weights", "pu", "pv", "cps")

    def __init__(self, surf) -> None:
        self.uflat = flat_knots(surf.uknots, surf.umults)
        self.vflat = flat_knots(surf.vknots, surf.vmults)
        self.weights = surf._weights
        self.pu = surf.udegree
        self.pv = surf.vdegree
        self.cps = surf.control_points

    def clamp(self, u: float, v: float) -> tuple:
        return (
            min(max(u, self.uflat[self.pu]), self.uflat[-self.pu - 1]),
            min(max(v, self.vflat[self.pv]), self.vflat[-self.pv - 1]),
        )

    def weight(self, i: int, j: int) -> float:
        if self.weights is None:
            return 1.0
        return self.weights[i][j]


def _data_of(surf) -> _Data:
    """The prepared payload, cached on the surface.

    Knot expansion is pure overhead per evaluation; surfaces are treated
    as values in this codebase (solvers build new ones), so caching on
    identity is safe. Mutating `control_points` in place bypasses it --
    don't.
    """
    cached = getattr(surf, "_npu_data", None)
    if cached is None:
        cached = _Data(surf)
        try:
            surf._npu_data = cached
        except AttributeError:
            pass
    return cached


def _accumulate(d: _Data, Nu: dict, Nv: dict):
    """Homogeneous point ⊕ weight sum for the given basis slices."""
    hpt = [0.0, 0.0, 0.0]
    wsum = 0.0
    for i, ni in Nu.items():
        row = d.cps[i]
        for j, nj in Nv.items():
            c = row[j]
            h = ni * nj * d.weight(i, j)
            hpt[0] += h * c.x
            hpt[1] += h * c.y
            hpt[2] += h * c.z
            wsum += h
    return hpt, wsum


def eval_point(surf, u: float, v: float) -> Vec3:
    """Evaluate the surface point at ``(u, v)`` (rational-safe homogeneous divide)."""
    return _eval_point_data(_data_of(surf), u, v)


def _eval_all_data(d: _Data, u: float, v: float) -> tuple:
    """Point + both tangents from a single basis build (prepared payload)."""
    u, v = d.clamp(u, v)
    Nu, dNu = _basis_and_deriv(u, d.pu, d.uflat)
    Nv, dNv = _basis_and_deriv(v, d.pv, d.vflat)

    hpt, wsum = _accumulate(d, Nu, Nv)
    if wsum == 0.0:
        raise ZeroDivisionError("surface weights sum to zero at (u, v)")
    px, py, pz = hpt[0] / wsum, hpt[1] / wsum, hpt[2] / wsum

    d_ho = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    d_w = [0.0, 0.0]
    for i, ni in Nu.items():
        row = d.cps[i]
        for j, nj in Nv.items():
            c = row[j]
            wij = d.weight(i, j)
            dNu_i = dNu.get(i, 0.0)
            dNv_j = dNv.get(j, 0.0)
            d_ho[0][0] += dNu_i * nj * wij * c.x
            d_ho[0][1] += dNu_i * nj * wij * c.y
            d_ho[0][2] += dNu_i * nj * wij * c.z
            d_w[0] += dNu_i * nj * wij
            d_ho[1][0] += ni * dNv_j * wij * c.x
            d_ho[1][1] += ni * dNv_j * wij * c.y
            d_ho[1][2] += ni * dNv_j * wij * c.z
            d_w[1] += ni * dNv_j * wij
    return (
        Vec3(px, py, pz),
        Vec3(
            (d_ho[0][0] - d_w[0] * px) / wsum,
            (d_ho[0][1] - d_w[0] * py) / wsum,
            (d_ho[0][2] - d_w[0] * pz) / wsum,
        ),
        Vec3(
            (d_ho[1][0] - d_w[1] * px) / wsum,
            (d_ho[1][1] - d_w[1] * py) / wsum,
            (d_ho[1][2] - d_w[1] * pz) / wsum,
        ),
    )


def eval_all(surf, u: float, v: float) -> tuple:
    """Point + both analytic tangents at ``(u, v)``, one basis build.

    The combined kernel behind the hot paths (projection, row assembly):
    one `_Data` lookup and one basis evaluation instead of three.
    """
    return _eval_all_data(_data_of(surf), u, v)


def _eval_point_data(d: _Data, u: float, v: float) -> Vec3:
    """Point-only evaluation on a prepared payload (the coarse-pass kernel)."""
    u, v = d.clamp(u, v)
    Nu = _nonzero_row(u, d.pu, d.uflat)
    Nv = _nonzero_row(v, d.pv, d.vflat)
    hpt, wsum = _accumulate(d, Nu, Nv)
    if wsum == 0.0:
        raise ZeroDivisionError("surface weights sum to zero at (u, v)")
    return Vec3(hpt[0] / wsum, hpt[1] / wsum, hpt[2] / wsum)


def eval_tangents(surf, u: float, v: float) -> tuple:
    """Analytic first-order derivatives ``(∂S/∂u, ∂S/∂v)`` at ``(u, v)``.

    Homogeneous accumulation + quotient rule:
    ``∂S = (∂H − ∂W·S) / W`` — identical for rational and non-rational
    (non-rational collapses to constant W).
    """
    _pt, su, sv = _eval_all_data(_data_of(surf), u, v)
    return su, sv


def eval_normal(surf, u: float, v: float) -> Vec3:
    """Unit surface normal at ``(u, v)`` (cross product of the analytic tangents)."""
    su, sv = eval_tangents(surf, u, v)
    nx = su.y * sv.z - su.z * sv.y
    ny = su.z * sv.x - su.x * sv.z
    nz = su.x * sv.y - su.y * sv.x
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    return Vec3(nx / n, ny / n, nz / n)


def project_point(surf, pt, m: int = 6, n: int = 6, iters: int = 40):
    """Nearest surface point to ``pt``: ``(proj, (u, v))`` (no OCC calls).

    Coarse grid (point evaluations only) seeds a few damped Gauss-Newton
    runs on the orthogonality equations, then each boundary isocurve is
    checked by golden-section -- the global nearest point may lie on the
    boundary when the query is outside the footprint, which domain
    clamping alone does not find.
    """
    d = _data_of(surf)
    u0, u1 = d.uflat[d.pu], d.uflat[-d.pu - 1]
    v0, v1 = d.vflat[d.pv], d.vflat[-d.pv - 1]

    def _dist2(q) -> float:
        return (q.x - pt.x) ** 2 + (q.y - pt.y) ** 2 + (q.z - pt.z) ** 2

    # -- coarse pass: point evaluations only, keep the best few cells -----
    coarse = []
    for i in range(m + 1):
        for j in range(n + 1):
            uu = u0 + (u1 - u0) * i / m
            vv = v0 + (v1 - v0) * j / n
            coarse.append((_dist2(_eval_point_data(d, uu, vv)), uu, vv))
    coarse.sort(key=lambda entry: entry[0])
    seeds: list = []
    for dist, uu, vv in coarse:
        if all(abs(uu - s_u) + abs(vv - s_v) > 0.5 * min(
            (u1 - u0) / max(m, 1), (v1 - v0) / max(n, 1)
        ) for s_u, s_v in seeds):
            seeds.append((uu, vv))
        if len(seeds) >= 3:
            break

    # -- damped Gauss-Newton on r.Su = r.Sv = 0 -----------------------------
    best = (coarse[0][0], _eval_point_data(d, coarse[0][1], coarse[0][2]),
            (coarse[0][1], coarse[0][2]))
    for uu, vv in seeds:
        dist, uu, vv = _newton(d, pt, uu, vv, u0, u1, v0, v1,
                               iters=min(max(iters, 4), 12))
        if dist < best[0]:
            best = (dist, _eval_point_data(d, uu, vv), (uu, vv))

    # -- boundary isocurves by golden-section --------------------------------
    for fixed, lo, hi, along_u in (
        (u0, v0, v1, False), (u1, v0, v1, False),
        (v0, u0, u1, True), (v1, u0, u1, True),
    ):
        dist, uu, vv = _edge_min(d, pt, fixed, lo, hi, along_u)
        if dist < best[0]:
            best = (dist, _eval_point_data(d, uu, vv), (uu, vv))
    return best[1], best[2]


def _newton(d: _Data, pt, uu: float, vv: float, u0: float, u1: float,
            v0: float, v1: float, iters: int) -> tuple:
    """Damped Gauss-Newton on the orthogonality equations from one seed.

    Returns ``(dist2, u, v)``. Terminates early when the gradient is tiny;
    halves the step while the residual grows (at most four halvings).
    """
    domain = 10.0 * ((u1 - u0) + (v1 - v0))
    q, su, sv = _eval_all_data(d, uu, vv)
    dist = (q.x - pt.x) ** 2 + (q.y - pt.y) ** 2 + (q.z - pt.z) ** 2
    for _ in range(max(iters, 1)):
        rx, ry, rz = q.x - pt.x, q.y - pt.y, q.z - pt.z
        gu = rx * su.x + ry * su.y + rz * su.z
        gv = rx * sv.x + ry * sv.y + rz * sv.z
        if abs(gu) + abs(gv) < 1e-10:
            break
        suu = su.x * su.x + su.y * su.y + su.z * su.z
        svv = sv.x * sv.x + sv.y * sv.y + sv.z * sv.z
        suv = su.x * sv.x + su.y * sv.y + su.z * sv.z
        det = suu * svv - suv * suv
        if abs(det) < 1e-20:
            break
        du = -(gu * svv - gv * suv) / det
        dv = -(gv * suu - gu * suv) / det
        if abs(du) + abs(dv) > domain:
            du *= 0.5
            dv *= 0.5
        step, accepted = 1.0, False
        for _ in range(5):
            trial_u = min(max(uu + step * du, u0), u1)
            trial_v = min(max(vv + step * dv, v0), v1)
            trial = _eval_point_data(d, trial_u, trial_v)
            trial_dist = ((trial.x - pt.x) ** 2 + (trial.y - pt.y) ** 2
                          + (trial.z - pt.z) ** 2)
            if trial_dist <= dist:
                uu, vv, dist, accepted = trial_u, trial_v, trial_dist, True
                break
            step *= 0.5
        if not accepted:
            break
        q, su, sv = _eval_all_data(d, uu, vv)
    return dist, uu, vv


def _edge_min(d: _Data, pt, fixed: float, lo: float, hi: float,
              along_u: bool) -> tuple:
    """Golden-section minimum of the distance over one boundary isocurve."""
    golden = (math.sqrt(5.0) - 1.0) / 2.0

    def _at(t: float) -> tuple:
        if along_u:
            return _eval_point_data(d, t, fixed)
        return _eval_point_data(d, fixed, t)

    def _dist2(t: float) -> float:
        q = _at(t)
        return (q.x - pt.x) ** 2 + (q.y - pt.y) ** 2 + (q.z - pt.z) ** 2

    a, b = lo, hi
    c, e = b - golden * (b - a), a + golden * (b - a)
    fc, fe = _dist2(c), _dist2(e)
    for _ in range(24):
        if fc < fe:
            b, fe = e, fc
            e = c
            c = b - golden * (b - a)
            fc = _dist2(c)
        else:
            a, fc = c, fe
            c = e
            e = a + golden * (b - a)
            fe = _dist2(e)
    t = 0.5 * (a + b)
    if along_u:
        return _dist2(t), t, fixed
    return _dist2(t), fixed, t
