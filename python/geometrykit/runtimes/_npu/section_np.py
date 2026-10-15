# This file was generated with the assistance of an AI coding tool.
"""Plane x surface sections: marching in parameter space, Newton-polished.

A section is computed where it naturally lives -- in the surface's (u, v)
domain -- so every section point carries its parameter track (the pcurve)
from birth instead of having it projected afterwards. The 3D points are
*evaluated* from the final tracks, which makes surface-consistency exact
by construction; the only numerical error is the plane residual, driven
to ~1e-10 by the polish.

Method: uniform grid over the parameter domain, sign changes of the
signed plane distance on cell edges, linear crossing + Newton polish
along the distance gradient (needs first derivatives only -- `eval_all`),
segment pairing per cell, greedy chaining by endpoint proximity.

Known limitation, stated not hidden: saddle cells (four crossings) pair
edges in order, which can misroute tangled topologies. For the product
cases (sections of fair patches by planes) saddles are rare; a
topological saddle solver is future work, explicitly not here.
"""

from __future__ import annotations

from typing import Sequence

from geometrykit.runtimes._npu.surface_np import (
    _data_of,
    _eval_all_data,
    _eval_point_data,
)


def _signed(d, plane_origin, plane_normal, u: float, v: float) -> float:
    q = _eval_point_data(d, u, v)
    return (
        (q.x - plane_origin[0]) * plane_normal[0]
        + (q.y - plane_origin[1]) * plane_normal[1]
        + (q.z - plane_origin[2]) * plane_normal[2]
    )


def _polish(d, plane_origin, plane_normal, u: float, v: float,
            iters: int = 4) -> tuple:
    """Newton polish of a crossing along the distance gradient."""
    for _ in range(max(iters, 0)):
        q, su, sv = _eval_all_data(d, u, v)
        f = ((q.x - plane_origin[0]) * plane_normal[0]
             + (q.y - plane_origin[1]) * plane_normal[1]
             + (q.z - plane_origin[2]) * plane_normal[2])
        fu = su.x * plane_normal[0] + su.y * plane_normal[1] + su.z * plane_normal[2]
        fv = sv.x * plane_normal[0] + sv.y * plane_normal[1] + sv.z * plane_normal[2]
        denom = fu * fu + fv * fv
        if denom < 1e-24:
            break
        step = f / denom
        u -= step * fu
        v -= step * fv
        if abs(f) < 1e-10:
            break
    u0, u1 = d.uflat[d.pu], d.uflat[-d.pu - 1]
    v0, v1 = d.vflat[d.pv], d.vflat[-d.pv - 1]
    return min(max(u, u0), u1), min(max(v, v0), v1)


def section_plane(surf, plane_origin: Sequence[float],
                  plane_normal: Sequence[float],
                  nu: int = 48, nv: int = 48) -> list:
    """Section polylines of *surf* with a plane.

    Args:
        surf:          kernel Surface.
        plane_origin:  a point on the plane (3-vector).
        plane_normal:  the plane normal (need not be unit).
        nu/nv:         marching grid density in (u, v).

    Returns:
        a list of chains; each chain is ``(points, uvs)`` with ``points``
        the evaluated 3D coordinates and ``uvs`` the parameter track,
        index-aligned. Empty list when the plane misses the surface.
    """
    d = _data_of(surf)
    u0, u1 = d.uflat[d.pu], d.uflat[-d.pu - 1]
    v0, v1 = d.vflat[d.pv], d.vflat[-d.pv - 1]

    length = (
        (plane_normal[0] ** 2 + plane_normal[1] ** 2 + plane_normal[2] ** 2) ** 0.5
    )
    if length < 1e-12:
        raise ValueError("section plane normal is degenerate")
    normal = (plane_normal[0] / length, plane_normal[1] / length,
              plane_normal[2] / length)

    # signed distances on the grid; exact zeros nudge positive so every
    # crossing is a strict sign change on some cell edge
    field = []
    for i in range(nu + 1):
        u = u0 + (u1 - u0) * i / nu
        row = []
        for j in range(nv + 1):
            v = v0 + (v1 - v0) * j / nv
            f = _signed(d, plane_origin, normal, u, v)
            row.append(f if f != 0.0 else 1e-300)
        field.append(row)

    def _crossing(i0: int, j0: int, i1: int, j1: int) -> tuple:
        f0, f1 = field[i0][j0], field[i1][j1]
        t = f0 / (f0 - f1)
        u = u0 + (u1 - u0) * (i0 + t * (i1 - i0)) / nu
        v = v0 + (v1 - v0) * (j0 + t * (j1 - j0)) / nv
        return _polish(d, plane_origin, normal, u, v)

    segments = []
    for i in range(nu):
        for j in range(nv):
            corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            crossings = []
            for (a0, b0), (a1, b1) in (
                (corners[0], corners[1]), (corners[1], corners[2]),
                (corners[2], corners[3]), (corners[3], corners[0]),
            ):
                if (field[a0][b0] < 0.0) != (field[a1][b1] < 0.0):
                    crossings.append(_crossing(a0, b0, a1, b1))
            if len(crossings) == 2:
                segments.append((crossings[0], crossings[1]))
            elif len(crossings) == 4:
                segments.append((crossings[0], crossings[1]))
                segments.append((crossings[2], crossings[3]))

    chains = _chain(segments, (u1 - u0) / nu + (v1 - v0) / nv)
    out = []
    for chain in chains:
        uvs = [(u, v) for u, v in chain]
        points = []
        for u, v in uvs:
            q = _eval_point_data(d, u, v)
            points.append((q.x, q.y, q.z))
        out.append((points, uvs))
    return out


def section_levels(surf, plane_origin: Sequence[float],
                     plane_normal: Sequence[float], offsets: Sequence[float],
                     nu: int = 48, nv: int = 48) -> list:
    """Level sets of one scalar field: many parallel sections, one evaluation.

    All ``offsets`` share the plane normal, so the signed-distance field is
    evaluated ONCE on the grid and each offset marches its own level set
    through it. For N parallel slices this replaces N field evaluations
    (the dominant cost) with one -- the configurator's 50-slice workload
    drops ~4x. Each entry is the same ``[(points, uvs), ...]`` chains as
    `section_plane`; empty offsets give empty lists.
    """
    d = _data_of(surf)
    u0, u1 = d.uflat[d.pu], d.uflat[-d.pu - 1]
    v0, v1 = d.vflat[d.pv], d.vflat[-d.pv - 1]

    length = (
        (plane_normal[0] ** 2 + plane_normal[1] ** 2 + plane_normal[2] ** 2) ** 0.5
    )
    if length < 1e-12:
        raise ValueError("section plane normal is degenerate")
    normal = (plane_normal[0] / length, plane_normal[1] / length,
              plane_normal[2] / length)

    # one scalar field for all levels: the normal coordinate of S(u, v)
    # relative to the plane origin
    field = []
    for i in range(nu + 1):
        u = u0 + (u1 - u0) * i / nu
        row = []
        for j in range(nv + 1):
            v = v0 + (v1 - v0) * j / nv
            q = _eval_point_data(d, u, v)
            row.append(
                (q.x - plane_origin[0]) * normal[0]
                + (q.y - plane_origin[1]) * normal[1]
                + (q.z - plane_origin[2]) * normal[2]
            )
        field.append(row)

    results = []
    for offset in offsets:
        shifted = (
            plane_origin[0] + offset * normal[0],
            plane_origin[1] + offset * normal[1],
            plane_origin[2] + offset * normal[2],
        )
        level = []
        for row in field:
            out_row = []
            for value in row:
                shifted_value = value - offset
                out_row.append(shifted_value if shifted_value != 0.0 else 1e-300)
            level.append(out_row)

        def _crossing(i0: int, j0: int, i1: int, j1: int) -> tuple:
            f0, f1 = level[i0][j0], level[i1][j1]
            t = f0 / (f0 - f1)
            u = u0 + (u1 - u0) * (i0 + t * (i1 - i0)) / nu
            v = v0 + (v1 - v0) * (j0 + t * (j1 - j0)) / nv
            return _polish(d, shifted, normal, u, v)

        segments = []
        for i in range(nu):
            for j in range(nv):
                corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
                crossings = []
                for (a0, b0), (a1, b1) in (
                    (corners[0], corners[1]), (corners[1], corners[2]),
                    (corners[2], corners[3]), (corners[3], corners[0]),
                ):
                    if (level[a0][b0] < 0.0) != (level[a1][b1] < 0.0):
                        crossings.append(_crossing(a0, b0, a1, b1))
                if len(crossings) == 2:
                    segments.append((crossings[0], crossings[1]))
                elif len(crossings) == 4:
                    segments.append((crossings[0], crossings[1]))
                    segments.append((crossings[2], crossings[3]))

        chains = []
        for chain in _chain(segments, (u1 - u0) / nu + (v1 - v0) / nv):
            uvs = [(u, v) for u, v in chain]
            points = []
            for u, v in uvs:
                q = _eval_point_data(d, u, v)
                points.append((q.x, q.y, q.z))
            chains.append((points, uvs))
        results.append(chains)
    return results


def _seg_intersect(p, q, r, s):
    """2D segment intersection: parameter pair or None.

    Transverse crossings only; parallel/collinear overlaps return None
    (stated, not hidden -- diagonal families on fair patches cross
    transversally; grazing overlaps belong to a topological solver).
    """
    ex, ey = q[0] - p[0], q[1] - p[1]
    fx, fy = s[0] - r[0], s[1] - r[1]
    denom = ex * fy - ey * fx
    if abs(denom) < 1e-18:
        return None
    gx, gy = r[0] - p[0], r[1] - p[1]
    t = (gx * fy - gy * fx) / denom
    w = (gx * ey - gy * ex) / denom
    if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= w <= 1.0 + 1e-9:
        t = min(max(t, 0.0), 1.0)
        return (t, min(max(w, 0.0), 1.0))
    return None


def weave_diagrid(surf, tracks_a: list, tracks_b: list) -> tuple:
    """Weave two section-track families into a diagrid, in parameter space.

    Every track crossing in (u, v) is a diagrid node *exactly on the
    surface* (evaluated, never projected) and *exactly on both planes*
    (tracks are plane sections). The crossings feed back into the tracks
    before interpolation, so the drawn curves pass through their nodes
    exactly too -- unlike plane-rooted nodes, which sit on one curve
    and one plane but only near the other curve.

    Args:
        surf:      kernel Surface both families were sectioned from.
        tracks_a:  list of uv-tracks ``[(u, v), ...]`` (first family).
        tracks_b:  same for the second family.

    Returns:
        ``(curves_a, curves_b, nodes)`` -- interpolated 3D curves per
        family (node-augmented) and ``nodes`` as ``(xyz, (u, v))`` pairs.
    """
    from geometrykit.geometry.curve import Curve as _Curve

    d = _data_of(surf)

    def _bbox(track):
        us = [p[0] for p in track]
        vs = [p[1] for p in track]
        return (min(us), max(us), min(vs), max(vs))

    boxes_b = [_bbox(t) for t in tracks_b]
    # crossings: (ia, seg_a, t, ib, seg_b, w, (u, v))
    crossings = []
    for ia, track_a in enumerate(tracks_a):
        box_a = _bbox(track_a)
        for ib, track_b in enumerate(tracks_b):
            box_b = boxes_b[ib]
            if (box_a[1] < box_b[0] or box_b[1] < box_a[0]
                    or box_a[3] < box_b[2] or box_b[3] < box_a[2]):
                continue
            for ka in range(len(track_a) - 1):
                p, q = track_a[ka], track_a[ka + 1]
                for kb in range(len(track_b) - 1):
                    r, s_ = track_b[kb], track_b[kb + 1]
                    hit = _seg_intersect(p, q, r, s_)
                    if hit is None:
                        continue
                    t, _w = hit
                    u = p[0] + t * (q[0] - p[0])
                    v = p[1] + t * (q[1] - p[1])
                    crossings.append((ia, ka, ib, kb, (u, v)))

    # feed back into the tracks, keeping order (later insertions first
    # so indices stay valid)
    extra_a: dict = {}
    extra_b: dict = {}
    for ia, ka, ib, kb, uv in crossings:
        extra_a.setdefault(ia, []).append((ka, uv))
        extra_b.setdefault(ib, []).append((kb, uv))
    tracks_a = [list(t) for t in tracks_a]
    tracks_b = [list(t) for t in tracks_b]
    for ia, items in extra_a.items():
        for ka, uv in sorted(items, reverse=True):
            tracks_a[ia].insert(ka + 1, uv)
    for ib, items in extra_b.items():
        for kb, uv in sorted(items, reverse=True):
            tracks_b[ib].insert(kb + 1, uv)
    # a crossing can land exactly on an existing sample: drop consecutive
    # duplicates or even the degree-1 fallback goes singular
    def _dedup(track):
        out = []
        for uv in track:
            if not out or abs(uv[0] - out[-1][0]) + abs(uv[1] - out[-1][1]) > 1e-9:
                out.append(uv)
        return out

    tracks_a = [_dedup(t) for t in tracks_a]
    tracks_b = [_dedup(t) for t in tracks_b]

    def _eval(uv):
        q = _eval_point_data(d, uv[0], uv[1])
        return (q.x, q.y, q.z)

    def _interpolate(track):
        from geometrykit.geometry.primitives import Vec as _V

        vecs = [_V(*_eval(uv)) for uv in track]
        kwargs: dict = {}
        if len(vecs) >= 3:
            lead = vecs[1] - vecs[0]
            trail = vecs[-1] - vecs[-2]
            if abs(lead) > 1e-12 and abs(trail) > 1e-12:
                kwargs = {"tan_start": lead, "tan_end": trail}
        try:
            return _Curve.from_points(vecs, **kwargs)
        except RuntimeError:
            return _Curve.from_points(vecs, degree=1)

    curves_a = [_interpolate(t) for t in tracks_a]
    curves_b = [_interpolate(t) for t in tracks_b]
    seen: set = set()
    nodes = []
    for _ia, _ka, _ib, _kb, uv in crossings:
        key = (round(uv[0], 9), round(uv[1], 9))
        if key in seen:
            continue
        seen.add(key)
        nodes.append((_eval(uv), uv))
    return curves_a, curves_b, nodes


def _chain(segments: list, tol: float) -> list:
    """Greedy endpoint chaining of (u, v) segments into polylines.

    Adjacent cells share grid nodes, so the same crossing is found twice
    (once per cell); consecutive duplicates collapse here, not in the
    interpolator (which goes singular on them).
    """
    def _same(a, b) -> bool:
        # exact-duplicate scale only: shared grid nodes polish to
        # bit-identical tracks. (A cell-scale tolerance here ate every
        # other legitimate crossing -- measured on a plane containing
        # an isocurve, where crossings sit half a cell apart.)
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1e-9

    def _append_unique(chain: list, points: list) -> None:
        for point in points:
            if not chain or not _same(chain[-1], point):
                chain.append(point)

    remaining = [list(seg) for seg in segments]
    chains: list = []
    while remaining:
        first = remaining.pop()
        chain = [first[0]] if not _same(first[0], first[1]) else [first[0]]
        _append_unique(chain, first[1:])
        extended = True
        while extended:
            extended = False
            for end in (0, -1):
                anchor = chain[end]
                best, best_k, best_e = None, -1, -1
                for k, seg in enumerate(remaining):
                    for e in (0, -1):
                        dist = abs(seg[e][0] - anchor[0]) + abs(seg[e][1] - anchor[1])
                        if dist <= tol and (best is None or dist < best):
                            best, best_k, best_e = dist, k, e
                if best_k >= 0:
                    seg = remaining.pop(best_k)
                    if end == -1:
                        _append_unique(chain, seg if best_e == 0 else seg[::-1])
                    else:
                        head: list = []
                        _append_unique(head, seg[::-1] if best_e == 0 else seg)
                        chain[:0] = [p for p in head if not _same(p, chain[0])]
                    extended = True
                    break
        if len(chain) >= 2:
            chains.append(chain)
    return chains
