# This file was generated with the assistance of an AI coding tool.
"""P-1: G1-constrained filling over the P-0 discrete Coons base.

The G1 constraint, made linear:
- at a boundary edge, the patch's first cross-derivative vector
  ``d = ∂P/∂cross`` must satisfy ``d ⋅ N̂(s) = 0`` where ``N̂(s)`` is the
  unit normal of the constraint surface (the ribbon) at the shared
  station — tangent-plane continuity with the constraint surface.
- for a clamped cubic in the cross direction, that derivative is
  ``3·(P[1] − P[0]) / span`` — linear in the control net. Adjusting
  control row 1 **along the normal direction** satisfies the constraint
  exactly without touching the magnitude contract:

      P'₁ = P₁ − (d ⋅ N̂) · N̂     (the off-plane component removed)

- cross-derivative *magnitude* is free (G1 is direction only) — the
  row-1 displacement along N̂ is the patch turning to meet the tangent
  plane.

Increment 1 scope (this file): one adjustment pass in the order
south → north → west → east, restricted to each edge's **interior
stations** so corner control points stay put (corner/edge interaction is
the P-2 datum-protocol increment). Normal fields are supplied as
precomputed samples per edge (see ``sample_ribbon_normal_field`` for the
OCCT-twin probe).
"""

from __future__ import annotations

import math
from typing import Sequence

from geometrykit.geometry.primitives import Vec
from geometrykit.runtimes._npu.patch_np import (
    discrete_coons_grid,
    discrete_coons_patch,
)


def sample_ribbon_normal_field(base_curve, ribbon_surface: "Surface", stations: int = 13) -> list:
    """Sample the ribbon surface's unit normal along the edge shared with
    *base_curve* (the OCCT-twin probe: nearest-boundary match on both
    ribbon long-edges, normal at that station).

    Returns ``[(vec3, param_record)]`` — the normal per interior station in
    base-curve parameter order, plus the ribbon param that matched.
    """
    out = []
    for k in range(stations):
        t = k / (stations - 1)
        q = base_curve.point_at(t)
        # probe both ribbon long-edges; the one lying on the base curve wins
        best = None
        for v_edge in (0.0, 1.0):
            ps = [ribbon_surface.point_at(i / (stations - 1), v_edge) for i in range(stations)]
            nearest = min(
                ((abs(pr.x - q.x) + abs(pr.y - q.y) + abs(pr.z - q.z), i) for i, pr in enumerate(ps)),
                key=lambda pair: pair[0],
            )
            dist, ui = nearest
            if best is None or dist < best[0]:
                n = ribbon_surface.normal_at(ui / (stations - 1), v_edge)
                best = (dist, n, ui / (stations - 1), v_edge)
        dist, n, u_s, v_s = best
        if dist > 1e-4:
            raise ValueError(
                f"ribbon surface does not share an edge with the base curve "
                f"(nearest approach {dist:.3e})"
            )
        ln = math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z)
        if ln == 0.0:
            raise ValueError("ribbon surface has a degenerate normal at the station")
        out.append((n.x / ln, n.y / ln, n.z / ln, u_s, v_s))
    return out


def _clamp_component_fix(p0, p1, normal_unit: tuple) -> Vec:
    """Move p1 so that (p1 − p0) ⊥ normal_unit (remove the off-plane component)."""
    d = (p1.x - p0.x, p1.y - p0.y, p1.z - p0.z)
    dn = d[0] * normal_unit[0] + d[1] * normal_unit[1] + d[2] * normal_unit[2]
    return Vec(
        p1.x - dn * normal_unit[0],
        p1.y - dn * normal_unit[1],
        p1.z - dn * normal_unit[2],
    )


def enforce_g1_edges(
    surface: "Surface",
    edge_normals: dict,
) -> "Surface":
    """P-1: adjust the boundary-adjacent control rows/cols so the patch's
    cross-derivative at each G1 edge is orthogonal to the edges' normal
    samples (tangent-plane continuity). Corner control points stay fixed.

    Args:
        surface:      the P-0 discrete Coons surface (mutated copy returned).
        edge_normals: ``{"south": [...], "north": [...], "west": [...],
                       "east": [...]}`` — interiors-only unit normals,
                       ordered along each edge's own parameter direction
                       (south/north: west→east; west/east: south→north).
                       An edge key absent → no G1 for that edge.

    Returns:
        A new ``Surface`` (the input is copied before mutation).
    """
    from geometrykit.geometry.surface import Surface as _Surface

    s = surface
    nu = len(s.control_points)       # control rows in u (0..nu−1), each nv long
    nv = len(s.control_points[0])    # control cols in v
    net = [[Vec(p.x, p.y, p.z) for p in row] for row in s.control_points]

    # v-direction knot-vspan matters only via direction (magnitude free) —
    # the adjustment is purely normal-directional, so no span math needed.

    def _fix_row1():
        normals = edge_normals.get("south")
        if normals is None:
            return
        for k, i in enumerate(range(1, nu - 1)):
            if k >= len(normals):
                break
            net[i][1] = _clamp_component_fix(net[i][0], net[i][1], normals[k])

    def _fix_row_north():
        normals = edge_normals.get("north")
        if normals is None:
            return
        for k, i in enumerate(range(1, nu - 1)):
            if k >= len(normals):
                break
            net[i][nv - 2] = _clamp_component_fix(net[i][nv - 1], net[i][nv - 2], normals[k])

    def _fix_col1():
        normals = edge_normals.get("west")
        if normals is None:
            return
        for k, j in enumerate(range(1, nv - 1)):
            if k >= len(normals):
                break
            net[1][j] = _clamp_component_fix(net[0][j], net[1][j], normals[k])

    def _fix_col_north_east():
        normals = edge_normals.get("east")
        if normals is None:
            return
        for k, j in enumerate(range(1, nv - 1)):
            if k >= len(normals):
                break
            net[nu - 2][j] = _clamp_component_fix(net[nu - 1][j], net[nu - 2][j], normals[k])

    _fix_row1()
    _fix_row_north()
    _fix_col1()
    _fix_col_north_east()

    return _Surface(
        control_points=net,
        uknots=list(s.uknots),
        vknots=list(s.vknots),
        umults=list(s.umults),
        vmults=list(s.vmults),
        udegree=s.udegree,
        vdegree=s.vdegree,
    )


def constrained_coons_patch(
    curves: Sequence,
    edge_normal_fields: dict,
    nu: int = 12,
    nv: int = 12,
    tol: float = 1e-6,
    degree: int = 3,
):
    """P-1 entry point: G1-constrained discrete Coons patch.

    Args:
        curves:             ``[south, east, north, west]`` (drawn directions
                            auto-repaired).
        edge_normal_fields: ``{"south": [...], "north": [...], "west": [...],
                            "east": [...]}`` — per-edge unit normal samples
                            (interiors only, ordered along each edge's own
                            parameter direction: south/north west→east;
                            west/east south→north). Fields from the ribbons
                            (OCCT-sampled today, datum-built after M6).
        nu/nv:              station counts for the P-0 base.
        tol:                corner coincidence tolerance.
        degree:             tensor-interpolation degree.

    Returns:
        ``(surface, corners)`` — G1-adjusted at the sampled stations.
    """
    surface, corners = discrete_coons_patch(curves, nu=nu, nv=nv, tol=tol, degree=degree)
    fixed = enforce_g1_edges(surface, edge_normal_fields)
    return fixed, corners
