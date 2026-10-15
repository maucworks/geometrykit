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

    # A near-grazing tangency makes the marching contour close a micro
    # loop at the very tail (measured on the roof: exit at v=0, re-entry,
    # exact return to the pre-exit sample). The linearized loop doubles
    # back and whips the interpolated end span (measured 8e-03
    # off-surface). Cut the excursion at its return point; index 0 is
    # exempt so genuinely closed island contours keep their ring.
    def _cut_tail_loop(track):
        n = len(track)
        for j in range(1, n - 2):
            if (abs(track[-1][0] - track[j][0]) + abs(track[-1][1] - track[j][1])
                    <= 1e-9):
                return track[:j + 1]
        return track

    tracks_a = [_cut_tail_loop(t) for t in tracks_a]
    tracks_b = [_cut_tail_loop(t) for t in tracks_b]

    def _bbox(track):
        us = [p[0] for p in track]
        vs = [p[1] for p in track]
        return (min(us), max(us), min(vs), max(vs))

    # crossings: (ia, seg_a, t, ib, seg_b, w, (u, v))
    #
    # Two diagonal families' track bboxes all overlap each other, so the
    # track-level short circuit alone leaves ~segments(A)*segments(B)
    # candidate pairs (~4.6M python touches on the shipped diagrid,
    # measured 845 ms in the weave). Index the B segments on a 2D bucket
    # grid instead: each A segment only tests the B segments whose
    # bounding boxes share a cell — bounded by the segments' own size,
    # not by the domain.
    _CELL = 0.05

    def _cells(seg):
        us = [min(p[0], q[0]), max(p[0], q[0])]
        vs = [min(p[1], q[1]), max(p[1], q[1])]
        if us[1] - us[0] >= _CELL or vs[1] - vs[0] >= _CELL:
            return None  # outlier-wide segment: callers fall back
        return [
            (cx, cy)
            for cx in range(int(us[0] / _CELL), int(us[1] / _CELL) + 1)
            for cy in range(int(vs[0] / _CELL), int(vs[1] / _CELL) + 1)
        ]

    index_b: dict = {}
    wide_b = []  # segments the bucket grid cannot bound tightly
    for ib, track_b in enumerate(tracks_b):
        for kb in range(len(track_b) - 1):
            p, q = track_b[kb], track_b[kb + 1]
            cells = _cells((p, q))
            if cells is None:
                wide_b.append((ib, kb, p, q))
                continue
            for cell in cells:
                index_b.setdefault(cell, []).append((ib, kb, p, q))

    touching = {}

    def _test_a(ia, ka, p, q):
        us = (min(p[0], q[0]), max(p[0], q[0]))
        vs = (min(p[1], q[1]), max(p[1], q[1]))
        seen: set = set()
        # A segment's own cell span may itself exceed _CELL at cut
        # entries; iterate the box anyway (worst case the old cost on
        # that one segment).
        for cx in range(int(us[0] / _CELL), int(us[1] / _CELL) + 1):
            for cy in range(int(vs[0] / _CELL), int(vs[1] / _CELL) + 1):
                for ib, kb, r, s_ in index_b.get((cx, cy), ()):
                    if (ib, kb) in seen:
                        continue
                    seen.add((ib, kb))
                    hit = _seg_intersect(p, q, r, s_)
                    if hit is None:
                        continue
                    t, _w = hit
                    u = p[0] + t * (q[0] - p[0])
                    v = p[1] + t * (q[1] - p[1])
                    touching.setdefault(ia, {}).setdefault(ka, []).append(
                        (ib, kb, (u, v))
                    )
        for ib, kb, r, s_ in wide_b:
            hit = _seg_intersect(p, q, r, s_)
            if hit is None:
                continue
            t, _w = hit
            u = p[0] + t * (q[0] - p[0])
            v = p[1] + t * (q[1] - p[1])
            touching.setdefault(ia, {}).setdefault(ka, []).append(
                (ib, kb, (u, v))
            )

    for ia, track_a in enumerate(tracks_a):
        for ka in range(len(track_a) - 1):
            p, q = track_a[ka], track_a[ka + 1]
            _test_a(ia, ka, p, q)

    crossings = []
    for ia, per_ka in touching.items():
        for ka, hits in per_ka.items():
            for ib, kb, (u, v) in hits:
                crossings.append((ia, ka, ib, kb, (u, v)))

    # feed back into the tracks, keeping order (later insertions first
    # so indices stay valid)
    extra_a: dict = {}
    extra_b: dict = {}
    for ia, ka, ib, kb, uv in crossings:
        extra_a.setdefault(ia, []).append((ka, uv))
        extra_b.setdefault(ib, []).append((kb, uv))
    tracks_a = [[(uv, False) for uv in t] for t in tracks_a]
    tracks_b = [[(uv, False) for uv in t] for t in tracks_b]
    for ia, items in extra_a.items():
        for ka, uv in sorted(items, reverse=True):
            tracks_a[ia].insert(ka + 1, (uv, True))
    for ib, items in extra_b.items():
        for kb, uv in sorted(items, reverse=True):
            tracks_b[ib].insert(kb + 1, (uv, True))
    # Thin to well-spaced 3D samples: dense global interpolation through
    # metrically clustered points goes ill-conditioned and the poles
    # explode -- measured 1e12 deviations from a 2.7e-07 m pair on a
    # 109-sample chain, and residual end wiggles even from a 5e-04 m
    # sample/node pair. Spacing is judged in meters (the chord
    # parameterization lives there, not in uv). Everything within 1 mm of
    # a kept point merges, and a node never loses its mark: when a node
    # lands on a kept sample it snaps *onto that slot* (the exact node
    # replaces the sample), so the node stays an interpolation point.
    def _eval(uv):
        q = _eval_point_data(d, uv[0], uv[1])
        return (q.x, q.y, q.z)

    def _thin(track):
        out = []  # [xyz, is_node]
        for uv, is_node in track:
            q = _eval(uv)  # nodes stay exact surface points
            if _near(q, out, 1e-3):
                if is_node and not out[-1][1]:
                    out[-1] = [q, True]  # snap node onto the sample slot
                continue
            out.append([q, is_node])
        return [q for q, _n in out]

    def _near(q, out, tol):
        return any(
            abs(q[0] - k[0]) + abs(q[1] - k[1]) + abs(q[2] - k[2]) <= tol
            for k, _n in out
        )

    tracks_a = [_thin(t) for t in tracks_a]
    tracks_b = [_thin(t) for t in tracks_b]

    def _interpolate(vecs):
        from geometrykit.geometry.primitives import Vec as _V

        vecs = [_V(*q) for q in vecs]
        # free ends: samples lie exactly on the surface, so the plain
        # chord-parameterised interpolation stays on it (measured 4e-07
        # off-surface on a 123-sample woven chain); end-tangent clamps --
        # unit or chord-scaled -- whip the end spans (measured 6e-03).
        try:
            return _Curve.from_points(vecs)
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
