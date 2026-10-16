# This file was generated with the assistance of an AI coding tool.
"""Adaptive kernel tessellation: OCC-free surface meshing.

A restricted quadtree over the parameter domain: quads split while
their geometric error exceeds the tolerances, neighbours are forced
to differ by at most one level, and split edges close with a fan from
the evaluated centroid (green closure). Every vertex is an evaluated
surface point and every edge is shared exactly, so the mesh is
watertight by construction; the only numerical content is the kernel
evaluation itself.

Error metric per quad: maximum edge-midpoint deviation (surface point
at the edge midpoint vs. the 3D edge midpoint) against ``deflection``,
and maximum corner-normal deviation against the quad normal versus
``angle_tol`` (radians). Stations are dyadic integer lattice points
``(iu, iv, level)`` with ``u = u0 + (u1-u0)*iu/2**level``, so shared
edges compute bit-identical vertices on both sides.
"""

from __future__ import annotations

import math

from geometrykit.runtimes._npu.surface_np import (
    _data_of,
    _eval_all_data,
)


def _station_uv(u0: float, u1: float, v0: float, v1: float,
                iu: int, iv: int, level: int) -> tuple:
    """Domain point of a dyadic lattice station."""
    scale = 1.0 / (1 << level)
    return (
        u0 + (u1 - u0) * iu * scale,
        v0 + (v1 - v0) * iv * scale,
    )


def adaptive_mesh(surf, deflection: float = 0.01, angle_tol: float = 0.5,
                  max_depth: int = 8, base_level: int = 2) -> tuple:
    """Triangulate *surf* adaptively; ``(verts, normals, tris)``.

    Args:
        surf:       kernel Surface.
        deflection: maximum distance of the surface from any triangle.
        angle_tol:  maximum normal deviation within a quad (radians).
        max_depth:  deepest quad level (base_level + refinements).
        base_level: starting grid is ``2**base_level`` quads per side.

    Returns:
        ``(verts, normals, tris, uvs)`` — ``verts`` flat ``[x, y, z]``,
        ``normals`` flat unit normals per vertex, ``tris`` index
        triples, ``uvs`` flat ``[u, v]`` per vertex (the parameter
        track, like section chains carry). Every edge is shared by
        exactly two triangles (watertight) on non-degenerate
        surfaces.
    """
    if deflection <= 0.0:
        raise ValueError(f"deflection must be positive, got {deflection}")
    if angle_tol <= 0.0:
        raise ValueError(f"angle_tol must be positive, got {angle_tol}")
    if max_depth < base_level:
        raise ValueError(
            f"max_depth {max_depth} below base_level {base_level}"
        )
    d = _data_of(surf)
    u0, u1 = d.uflat[d.pu], d.uflat[-d.pu - 1]
    v0, v1 = d.vflat[d.pv], d.vflat[-d.pv - 1]
    if not (u1 > u0 and v1 > v0):
        raise ValueError("degenerate parameter domain")

    # eval cache: (iu, iv, level) -> (point, normal), one basis build
    cache: dict = {}

    def _eval(iu: int, iv: int, level: int) -> tuple:
        key = (iu, iv, level)
        hit = cache.get(key)
        if hit is None:
            u, v = _station_uv(u0, u1, v0, v1, iu, iv, level)
            pt, su, sv = _eval_all_data(d, u, v)
            nx = su.y * sv.z - su.z * sv.y
            ny = su.z * sv.x - su.x * sv.z
            nz = su.x * sv.y - su.y * sv.x
            n = math.sqrt(nx * nx + ny * ny + nz * nz)
            if n > 0.0:
                nx, ny, nz = nx / n, ny / n, nz / n
            hit = ((pt.x, pt.y, pt.z), (nx, ny, nz))
            cache[key] = hit
        return hit

    def _corners(i: int, j: int, level: int) -> list:
        return [(i, j, level), (i + 1, j, level),
                (i + 1, j + 1, level), (i, j + 1, level)]

    def _mid(a: tuple, b: tuple) -> tuple:
        (aiu, aiv, al), (biu, biv, bl) = a, b
        assert al == bl, "midpoint of mixed-level stations"
        return (aiu + biu, aiv + biv, al + 1)

    def _quad_error(i: int, j: int, level: int) -> tuple:
        """(edge deviation, normal deviation) of one quad.

        Edge deviation covers quad edges, both diagonals and the
        centroid spoke deviations, so every emitted triangle edge is
        bounded (green-closure fans introduce diagonals and spokes
        that edge-only checks would miss).
        """
        corners = _corners(i, j, level)
        pts = [_eval(*c)[0] for c in corners]
        norms = [_eval(*c)[1] for c in corners]
        edge_dev = 0.0
        for k in range(4):
            a, b = corners[k], corners[(k + 1) % 4]
            mid = _eval(*_mid(a, b))[0]
            pa, pb = pts[k], pts[(k + 1) % 4]
            edge_dev = max(edge_dev, abs(mid[0] - 0.5 * (pa[0] + pb[0]))
                           + abs(mid[1] - 0.5 * (pa[1] + pb[1]))
                           + abs(mid[2] - 0.5 * (pa[2] + pb[2])))
        # diagonals and centroid (fan spokes land here)
        cu = (corners[0][0] + corners[2][0], corners[0][1] + corners[2][1],
              corners[0][2] + 1)
        cc = _eval(*cu)[0]
        for pa, pb in ((pts[0], pts[2]), (pts[1], pts[3])):
            edge_dev = max(edge_dev, abs(cc[0] - 0.5 * (pa[0] + pb[0]))
                           + abs(cc[1] - 0.5 * (pa[1] + pb[1]))
                           + abs(cc[2] - 0.5 * (pa[2] + pb[2])))
        bil = tuple(sum(p[k] for p in pts) / 4.0 for k in range(3))
        edge_dev = max(edge_dev, abs(cc[0] - bil[0])
                       + abs(cc[1] - bil[1]) + abs(cc[2] - bil[2]))
        # fan spokes: the green-closure fan connects the centroid to
        # every boundary station, so bound every spoke midpoint (covers
        # all split patterns; edge midpoints already evaluated above)
        cstat = (corners[0][0] + corners[2][0], corners[0][1] + corners[2][1],
                 corners[0][2] + 1)
        boundary = list(corners)
        for k in range(4):
            a, b = corners[k], corners[(k + 1) % 4]
            boundary.append(_mid(a, b))
        for (biu, biv, bl) in boundary:
            # centroid lives at level L+1, boundary at L or L+1:
            # lift the boundary numerator before halving
            lift = 1 << (cstat[2] - bl)
            mstat = (cstat[0] + lift * biu, cstat[1] + lift * biv,
                     cstat[2] + 1)
            sm = _eval(*mstat)[0]
            bp = _eval(biu, biv, bl)[0]
            edge_dev = max(edge_dev, abs(sm[0] - 0.5 * (cc[0] + bp[0]))
                           + abs(sm[1] - 0.5 * (cc[1] + bp[1]))
                           + abs(sm[2] - 0.5 * (cc[2] + bp[2])))
        # quad normal from the diagonals
        d0 = (pts[2][0] - pts[0][0], pts[2][1] - pts[0][1], pts[2][2] - pts[0][2])
        d1 = (pts[3][0] - pts[1][0], pts[3][1] - pts[1][1], pts[3][2] - pts[1][2])
        qn = (d0[1] * d1[2] - d0[2] * d1[1],
              d0[2] * d1[0] - d0[0] * d1[2],
              d0[0] * d1[1] - d0[1] * d1[0])
        ql = math.sqrt(qn[0] ** 2 + qn[1] ** 2 + qn[2] ** 2)
        normal_dev = 0.0
        if ql > 0.0:
            qn = (qn[0] / ql, qn[1] / ql, qn[2] / ql)
            for n in norms:
                cos_a = max(-1.0, min(1.0, n[0] * qn[0] + n[1] * qn[1] + n[2] * qn[2]))
                normal_dev = max(normal_dev, math.acos(cos_a))
        return edge_dev, normal_dev

    # leaves: (i, j, level) quads, base grid first
    n_base = 1 << base_level
    leaves: dict = {
        (i, j, base_level): None
        for i in range(n_base) for j in range(n_base)
    }

    def _neighbors(i: int, j: int, level: int) -> list:
        """Candidate edge-neighbours at levels level-1..level+1."""
        found = []
        for key in ((i - 1, j, level), (i + 1, j, level),
                    (i, j - 1, level), (i, j + 1, level),
                    (i >> 1, j >> 1, level - 1)
                    if level > 0 else None):
            if key is not None and key in leaves and key not in found:
                found.append(key)
        for key in ((2 * i - 1, 2 * j, level + 1),
                    (2 * i + 2, 2 * j, level + 1),
                    (2 * i - 1, 2 * j + 1, level + 1),
                    (2 * i + 2, 2 * j + 1, level + 1),
                    (2 * i, 2 * j - 1, level + 1),
                    (2 * i + 1, 2 * j - 1, level + 1),
                    (2 * i, 2 * j + 2, level + 1),
                    (2 * i + 1, 2 * j + 2, level + 1)):
            if key in leaves and key not in found:
                found.append(key)
        return found

    # refine to fixpoint: error splits, then balance (2:1 at most)
    while True:
        marked = set()
        for key in sorted(leaves):
            i, j, level = key
            if level >= max_depth:
                continue
            edge_dev, normal_dev = _quad_error(i, j, level)
            if edge_dev > deflection or normal_dev > angle_tol:
                marked.add(key)
        if not marked:
            break
        # balance: no edge-neighbour more than one level coarser
        while True:
            extra = set()
            for key in sorted(leaves):
                if key in marked:
                    continue
                i, j, level = key
                for nb in _neighbors(i, j, level):
                    if nb in marked:
                        continue
                    if nb[2] < level - 1 and nb[2] < max_depth:
                        # a marked finer quad touches this one: split up
                        if any(m[2] > nb[2] + 1 for m in marked
                               if _quads_touch(nb, m)):
                            extra.add(nb)
            if not extra:
                break
            marked |= extra
        for key in marked:
            i, j, level = key
            del leaves[key]
            leaves[(2 * i, 2 * j, level + 1)] = None
            leaves[(2 * i + 1, 2 * j, level + 1)] = None
            leaves[(2 * i, 2 * j + 1, level + 1)] = None
            leaves[(2 * i + 1, 2 * j + 1, level + 1)] = None

    # triangulate with green closure: split edges fan from the
    # evaluated centroid; every vertex is on the surface
    verts: list = []
    normals: list = []
    uvs: list = []
    index_of: dict = {}

    def _vertex(station: tuple) -> int:
        idx = index_of.get(station)
        if idx is None:
            (px, py, pz), (nx, ny, nz) = _eval(*station)
            idx = len(verts) // 3
            verts.extend((px, py, pz))
            normals.extend((nx, ny, nz))
            u, v = _station_uv(u0, u1, v0, v1, *station)
            uvs.extend((u, v))
            index_of[station] = idx
        return idx

    def _covers(candidate: tuple, a: tuple, b: tuple, level: int) -> bool:
        """True when the level-(level+1) quad covers part of edge a-b."""
        (ni, nj, _nl) = candidate
        (aiu, aiv, _al), (biu, biv, _bl) = a, b
        if aiu == biu:  # vertical edge
            eu = 2 * aiu
            lo, hi = sorted((2 * aiv, 2 * biv))
            return (ni == eu or ni + 1 == eu) and nj < hi and nj + 1 > lo
        # horizontal edge
        ev = 2 * aiv
        lo, hi = sorted((2 * aiu, 2 * biu))
        return (nj == ev or nj + 1 == ev) and ni < hi and ni + 1 > lo

    tris: list = []
    for key in sorted(leaves):
        i, j, level = key
        corners = _corners(i, j, level)
        finer = [nb for nb in _neighbors(i, j, level) if nb[2] == level + 1]
        loop = [corners[0]]
        for k in range(4):
            a, b = corners[k], corners[(k + 1) % 4]
            if any(_covers(nb, a, b, level) for nb in finer):
                loop.append(_mid(a, b))
            if k < 3:
                loop.append(b)
        if len(loop) == 4:
            a, b, c, e = (_vertex(p) for p in loop)
            tris.extend(((a, b, c), (a, c, e)))
        else:
            center = (corners[0][0] + corners[2][0],
                      corners[0][1] + corners[2][1], corners[0][2] + 1)
            c = _vertex(center)
            ring = [_vertex(p) for p in loop]
            for k in range(len(ring)):
                tris.append((c, ring[k], ring[(k + 1) % len(ring)]))
    return verts, normals, tris, uvs


def _quads_touch(a: tuple, b: tuple) -> bool:
    """True when quads a and b share an edge segment."""
    (ai, aj, al), (bi, bj, bl) = a, b
    scale = 1 << max(al, bl)
    au0, au1 = ai * (scale >> al), (ai + 1) * (scale >> al)
    av0, av1 = aj * (scale >> al), (aj + 1) * (scale >> al)
    bu0, bu1 = bi * (scale >> bl), (bi + 1) * (scale >> bl)
    bv0, bv1 = bj * (scale >> bl), (bj + 1) * (scale >> bl)
    share_v = (au1 == bu0 or bu1 == au0) and av0 < bv1 and bv0 < av1
    share_h = (av1 == bv0 or bv1 == av0) and au0 < bu1 and bu0 < au1
    return share_v or share_h
