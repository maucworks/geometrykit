# This file was generated with the assistance of an AI coding tool.
"""M8 prototype: tolerance-driven B-spline approximation over a solved field.

The exact-interpolation path maps the solved fill grid onto a dense clamped
net one-to-one with the sample stations. CAD-style representation instead
re-fits with far fewer knots, forgives detail within a tolerance, and joins
segments at C1. This module is the first step of that: one global
least-squares net over the field with an *adaptive knot vector*, segments
emerging from dichotomy cutting -- the structure OCCT's own approximation
cut, at a fraction of the machinery.

Boundary exactness is not weighted but *constrained*: the fit solves a KKT
system whose equality rows pin the four boundary station lines to the
field's own boundary stations, so the station map and its invariants
survive the representation change. First derivatives at the boundaries are
NOT constrained here; the G1/G2 row passes on the fitted surface restore
them, exactly as they do on the interpolated base today.

The segmentation bookkeeping is deliberately crude (knot insertion by
re-fitting, no Boehm updates): each knot-vector candidate re-runs the
whole least-squares solve, which is cheap at these sizes and keeps the
numeric path single.
"""

from __future__ import annotations

from typing import Sequence

from geometrykit.geometry.primitives import Vec
from geometrykit.geometry.surface import Surface
from geometrykit.runtimes._npu.curve_np import _nonzero_basis


def _expand(knots: Sequence[float], mults: Sequence[int]) -> list:
    flat = []
    for k, m in zip(knots, mults):
        flat.extend([k] * m)
    return flat


def _basis_at(t: float, p: int, flat, count: int) -> list:
    """Non-zero basis pairs (index, value); endpoints degrade to singleton."""
    if t <= 0.0:
        return [(0, 1.0)]
    if t >= 1.0:
        return [(count - 1, 1.0)]
    return sorted(_nonzero_basis(t, p, flat).items())


def _segments(flat: list) -> list:
    """Non-degenerate spans of a flat knot vector, as parameter pairs."""
    return [
        (flat[k], flat[k + 1])
        for k in range(len(flat) - 1)
        if flat[k + 1] - flat[k] > 1e-12
    ]


class NoFitSpace(ValueError):
    """The field gave the fitter too little to sample or too few poles."""


def adaptive_fit(
    field,
    degree: int = 4,
    tol: float = 1e-3,
    samples: tuple = (44, 44),
    max_segments: int = 8,
    ridge: float = 1e-9,
    stations: tuple = (None, None),
) -> tuple:
    """Fit `field` (a Surface) to `tol` with a C1-segmented net.

    ``stations`` is ``(u_params, v_params)`` -- the field's own boundary
    station parameters (the same lists the patch level passes to its row
    passes). The equalities pin the four station lines to the field at
    *exactly* those parameters, so the M0 invariant (the surface hits the
    boundary curves at the stations) survives the representation change.

    Returns ``(surface, report)``. The report carries the pole counts, the
    achieved sample residual and the knot structure, so callers can judge
    the compression honestly instead of trusting the tolerance.
    """
    p = int(degree)
    if p < 1:
        raise NoFitSpace("degree below 1 carries no segments")
    np = _numpy()
    nu_s, nv_s = samples
    us = [i / (nu_s - 1) for i in range(nu_s)]
    vs = [j / (nv_s - 1) for j in range(nv_s)]
    targets = np.array(
        [[_xyz(field.point_at(u, v)) for v in vs] for u in us]
    )
    station_us, station_vs = stations
    station_us = list(station_us) if station_us is not None else us
    station_vs = list(station_vs) if station_vs is not None else vs

    uknots, umults = [0.0, 1.0], [p + 1, p + 1]
    vknots, vmults = [0.0, 1.0], [p + 1, p + 1]

    report: dict = {"cuts": 0, "tol": tol, "degree": p}
    best = None  # (key, surface, snapshot); the cut path is not monotone
    while True:
        surface, residual, seg_u, seg_v = _fit_once(
            field, targets, us, vs, station_us, station_vs,
            uknots, umults, vknots, vmults, p, ridge
        )
        worst = max(max(seg_u, default=0.0), max(seg_v, default=0.0))
        segs_u = len(_segments(_expand(uknots, umults)))
        segs_v = len(_segments(_expand(vknots, vmults)))
        boundary = residual["boundary"]
        report["sample_worst"] = worst
        report["boundary_worst"] = boundary
        report["uknots"], report["umults"] = list(uknots), list(umults)
        report["vknots"], report["vmults"] = list(vknots), list(vmults)
        report["segments"] = (segs_u, segs_v)
        key = (boundary <= 1e-10, -worst)
        if best is None or key > best[0]:
            best = (key, surface, {
                "sample_worst": worst, "boundary_worst": boundary,
                "uknots": list(uknots), "umults": list(umults),
                "vknots": list(vknots), "vmults": list(vmults),
                "segments": (segs_u, segs_v), "cuts": report["cuts"],
            })
        # two gates: the sample tolerance AND the boundary contract. An
        # equality system that is infeasible at the current knot vector
        # shows up as a large equality residual even when the sample
        # residual is fine -- the ring row needs more poles before the
        # stations can be interpolated at all.
        if (
            (worst <= tol and boundary <= 1e-10)
            or segs_u >= max_segments
            or segs_v >= max_segments
        ):
            break
        cut_u = _cut_knots(uknots, umults, p, seg_u, tol)
        cut_v = _cut_knots(vknots, vmults, p, seg_v, tol)
        if boundary > 1e-10 and (not cut_u or not cut_v):
            # boundary infeasible: keep cutting BOTH directions even where
            # the sample residual would rest, until the stations fit
            cut_u = cut_u or _cut_knots(uknots, umults, p, seg_u, -1.0)
            cut_v = cut_v or _cut_knots(vknots, vmults, p, seg_v, -1.0)
        if not cut_u and not cut_v:
            break
        uknots, umults = cut_u
        vknots, vmults = cut_v
        report["cuts"] += 1
        if report["cuts"] > 2 * max_segments:
            break
    _key, best_surface, snap = best
    report.update(snap)
    return best_surface, report


def _cut_knots(knots, mults, p, seg_errors, tol):
    """Insert midpoint knots (C1 multiplicity) under the worst segments.

    Per cut at most four segments per direction, worst first, so the knot
    vector grows under observation instead of exploding.
    """
    flat = _expand(knots, mults)
    if not seg_errors or max(seg_errors) <= tol:
        return None
    segs = _segments(flat)
    ranked = sorted(range(len(seg_errors)), key=lambda k: -seg_errors[k])
    cut = []
    for k in ranked:
        if seg_errors[k] <= tol:
            break
        cut.append(round(0.5 * (segs[k][0] + segs[k][1]), 9))
        if len(cut) >= 4:
            break
    if not cut:
        return None
    merged = sorted(set(round(k, 9) for k in knots) | set(cut))
    inner = []
    for value in merged:
        if value <= 0.0 or value >= 1.0:
            continue
        # every interior knot stays at C1 multiplicity: a re-cut of an
        # existing node must not raise it toward C0 (that was the first
        # cut's whole point)
        inner.append(p - 1)
    return merged, [p + 1] + inner + [p + 1]


def _xyz(v):
    return (v.x, v.y, v.z)


def _numpy():
    """numpy, imported late so the pure helpers stay importable without it."""
    try:
        import numpy as np
    except ImportError:
        raise NoFitSpace(
            "adaptive_fit needs numpy (pip install numpy, or use the oracle env)"
        )
    return np


def _fit_once(field, targets, us, vs, station_us, station_vs,
              uknots, umults, vknots, vmults, p, ridge):
    """One least-squares solve over a knot vector, with boundary equalities."""
    np = _numpy()
    flat_u = _expand(uknots, umults)
    flat_v = _expand(vknots, vmults)
    n_cp_u = sum(umults) - (p + 1)
    n_cp_v = sum(vmults) - (p + 1)
    if n_cp_u < p + 1 or n_cp_v < p + 1:
        raise NoFitSpace("knot vector smaller than the degree demands")
    n = n_cp_u * n_cp_v

    samples = [
        (u, v, i, j)
        for i, u in enumerate(us)
        for j, v in enumerate(vs)
    ]
    rows, targets_m = [], []
    for u, v, i, j in samples:
        bu = _basis_at(u, p, flat_u, n_cp_u)
        bv = _basis_at(v, p, flat_v, n_cp_v)
        row = np.zeros(n)
        for iu, wu in bu:
            for iv, wv in bv:
                row[iu * n_cp_v + iv] += wu * wv
        rows.append(row)
        targets_m.append(targets[i][j])

    # boundary equalities: the four station lines, at the *station*
    # parameters -- the points the boundary-invariant is measured on. The
    # corners appear from two edges with consistent targets, so dedup.
    ring = []
    for t in station_us:
        ring.append((t, 0.0, _xyz(field.point_at(t, 0.0))))
        ring.append((t, 1.0, _xyz(field.point_at(t, 1.0))))
    for t in station_vs:
        ring.append((0.0, t, _xyz(field.point_at(0.0, t))))
        ring.append((1.0, t, _xyz(field.point_at(1.0, t))))
    equalities = []
    dupes = set()
    for u, v, target in ring:
        key = (round(u, 9), round(v, 9))
        if key in dupes:
            continue
        dupes.add(key)
        bu = _basis_at(u, p, flat_u, n_cp_u)
        bv = _basis_at(v, p, flat_v, n_cp_v)
        row = np.zeros(n)
        for iu, wu in bu:
            for iv, wv in bv:
                row[iu * n_cp_v + iv] += wu * wv
        equalities.append((row, target))

    A = np.array(rows)                      # (m, n)
    B = np.array(targets_m)                 # (m, 3)
    C = np.array([eq[0] for eq in equalities])
    D = np.array([eq[1] for eq in equalities])  # (m_eq, 3)

    # Equality-eliminated solve: x = x_particular + Z y, with Z the null
    # space of the boundary equalities. This makes station-boundary
    # exactness a property of the formulation rather than of the solver,
    # which a dense KKT solve does not honour numerically (measured 8.7e-5
    # through np.linalg.solve on this system).
    m_eq = C.shape[0] if C.size else 0
    if m_eq:
        _u, _s, vt = np.linalg.svd(C, full_matrices=True)
        rank = int((np.asarray(_s) > 1e-10 * max(float(_s.max()), 1.0)).sum())
        Z = vt[rank:].T                       # (n, n - rank)
        x_p = np.linalg.pinv(C) @ D           # (n, 3), min-norm particular
    else:
        Z = np.eye(n)
        x_p = np.zeros((n, 3))
    AtZ = A @ Z
    scale = max(1.0, float(np.trace(AtZ.T @ AtZ)) / max(Z.shape[1], 1))
    red = AtZ.T @ AtZ + np.eye(Z.shape[1]) * (ridge * scale)
    red_rhs = AtZ.T @ (B - A @ x_p)
    y = np.linalg.solve(red, red_rhs)
    try:
        y = np.linalg.solve(red, red_rhs)
    except np.linalg.LinAlgError:
        y = np.linalg.lstsq(red, red_rhs, rcond=None)[0]
    net_full = x_p + Z @ y
    net = net_full.reshape(n_cp_u, n_cp_v, 3)

    boundary_worst = (
        float(np.max(np.abs(C @ net_full - D))) if m_eq else 0.0
    )

    worst = 0.0
    seg_errors_u = [0.0] * len(_segments(flat_u))
    seg_errors_v = [0.0] * len(_segments(flat_v))
    segs_u = _segments(flat_u)
    segs_v = _segments(flat_v)
    for (u, v, i, j) in samples:
        bu = _basis_at(u, p, flat_u, n_cp_u)
        bv = _basis_at(v, p, flat_v, n_cp_v)
        acc = np.zeros(3)
        for iu, wu in bu:
            for iv, wv in bv:
                acc = acc + wu * wv * net[iu][iv]
        err = float(np.max(np.abs(acc - targets[i][j])))
        for k, (s0, s1) in enumerate(segs_u):
            if s0 - 1e-12 <= u <= s1 + 1e-12:
                seg_errors_u[k] = max(seg_errors_u[k], err)
        for k, (s0, s1) in enumerate(segs_v):
            if s0 - 1e-12 <= v <= s1 + 1e-12:
                seg_errors_v[k] = max(seg_errors_v[k], err)
        worst = max(worst, err)

    surface = _to_surface(net, uknots, umults, vknots, vmults, p)
    residual = {"sample_worst": worst, "boundary": boundary_worst}
    return surface, residual, seg_errors_u, seg_errors_v


def _to_surface(net, uknots, umults, vknots, vmults, p):
    rows = []
    for i in range(net.shape[0]):
        rows.append([Vec(*net[i][j]) for j in range(net.shape[1])])
    return Surface(
        control_points=rows,
        uknots=list(uknots),
        vknots=list(vknots),
        umults=list(umults),
        vmults=list(vmults),
        udegree=p,
        vdegree=p,
    )
