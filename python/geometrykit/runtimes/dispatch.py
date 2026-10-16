# This file was generated with the assistance of an AI coding tool.
"""Kernel dispatch: one API, engine-selected implementations.

The numpy mirror (`runtimes/_npu`) stays the API floor and the
reference; the `_rust` extension mirrors these functions bit-for-bit
(enforced by the parity harness). Dispatch routes the four first-order
kernel entries per process via `engine.engine()` — `npu` is always
available, `rust` when the extension is importable.

Error contract is part of the mirror: a zero weight sum raises
`ZeroDivisionError` from either engine (core returns the same message;
the wrapper maps it deliberately, no Rust panic crosses Python).
"""

from __future__ import annotations

import math


def _rust() -> object | None:
    from geometrykit import engine as _engine

    if _engine.engine() != "rust":
        return None
    from geometrykit import _rust

    return _rust


def basis_row_list(degree: int, uflat, stations):
    """`[{id: N}, ...]` per station, in order (engine-batched).

    Mirror `curve_np._nonzero_basis` under the active engine; dict
    insertion order is the mirror's (increasing id), as `.items()`
    consumers rely on it.
    """
    accel = _rust()
    if accel is None:
        from geometrykit.runtimes._npu import curve_np

        return [curve_np._nonzero_basis(t, degree, uflat) for t in stations]
    from geometrykit import _rust as _ext

    ids, vals = _ext.basis_row_grid(list(uflat), int(degree), [
        float(t) for t in stations
    ])
    stride = int(degree) + 1
    return [
        {ids[k] + j: vals[k * stride + j] for j in range(stride)}
        for k in range(len(stations))
    ]


def tessellate(surf, deflection: float = 0.01, angle_tol: float = 0.5,
               max_depth: int = 8, base_level: int = 2):
    """Mirror `tessellate_np.adaptive_mesh` under the active engine.

    Returns `(verts, normals, tris, uvs)` — flat xyz vertices, flat
    unit normals, flat index triples, flat uv tracks. Bit-identical
    across engines by harness.
    """
    from geometrykit.runtimes._npu import tessellate_np

    accel = _rust()
    if accel is None:
        return tessellate_np.adaptive_mesh(
            surf, deflection=deflection, angle_tol=angle_tol,
            max_depth=max_depth, base_level=base_level,
        )
    from geometrykit import _rust as _ext

    verts, normals, flat_tris, uvs = _ext.tessellate_grid(
        surf, float(deflection), float(angle_tol), int(max_depth),
        int(base_level),
    )
    tris = [
        (flat_tris[k], flat_tris[k + 1], flat_tris[k + 2])
        for k in range(0, len(flat_tris), 3)
    ]
    return verts, normals, tris, uvs


def plane_section_chains(surf, origin, normal, offsets, nu: int, nv: int):
    """Mirror `section_np.section_levels` under the active engine.

    Returns `[(points, uvs), ...]` per offset — chains of evaluated 3D
    points and their parameter tracks, index-aligned, exactly as the
    mirror's lists. Empty offsets give empty chains.
    """
    from geometrykit.runtimes._npu import section_np

    stream = _rust()
    if stream is None:
        return section_np.section_levels(surf, origin, normal, offsets, nu=nu, nv=nv)
    from geometrykit import _rust as _ext

    flat_out = _ext.section_levels_grid(
        surf,
        [origin[0], origin[1], origin[2]],
        [normal[0], normal[1], normal[2]],
        list(offsets), int(nu), int(nv),
    )
    pos = 0
    n_off = int(flat_out[pos])
    pos += 1
    results = []
    for _ in range(n_off):
        n_chains = int(flat_out[pos])
        pos += 1
        chains = []
        for _ in range(n_chains):
            m = int(flat_out[pos])
            pos += 1
            uvs = [
                (flat_out[pos + 2 * k], flat_out[pos + 2 * k + 1])
                for k in range(m)
            ]
            pos += 2 * m
            pts = []
            for k in range(m):
                pts.append((
                    flat_out[pos + 3 * k],
                    flat_out[pos + 3 * k + 1],
                    flat_out[pos + 3 * k + 2],
                ))
            pos += 3 * m
            chains.append((pts, uvs))
        results.append(chains)
    return results


def interpolate(points, degree: int = 3, tan_start=None, tan_end=None, knots_param=None):
    """Mirror `curve_np.interpolate` under the active engine.

    Returns `(control_points, knots, multiplicities)` in the mirror's
    tuples form, so callers can assemble the same `Curve`.
    """
    from geometrykit.runtimes._npu import curve_np

    accel = _rust()
    if accel is None:
        return curve_np.interpolate(
            points, degree=degree, tan_start=tan_start, tan_end=tan_end,
            knots_param=knots_param,
        )
    from geometrykit import _rust as _ext

    flat = []
    for p in points:
        flat.extend((p.x, p.y, p.z))
    ts = None if tan_start is None else [tan_start.x, tan_start.y, tan_start.z]
    te = None if tan_end is None else [tan_end.x, tan_end.y, tan_end.z]
    cpts_flat, knots, mults = _ext.interpolate_grid(
        flat, int(degree), ts, te,
        None if knots_param is None else list(knots_param),
    )
    cpts = [
        (cpts_flat[i], cpts_flat[i + 1], cpts_flat[i + 2])
        for i in range(0, len(cpts_flat), 3)
    ]
    return cpts, knots, mults


def banded_cholesky(rows, n: int, bandwidth: int):
    """Mirror `linalg_np.banded_cholesky` under the active engine.

    `rows[i]` is `{j: value}` as the mirror expects; the extension
    marshals each row to flat pairs, and the factor comes back in the
    mirror's own list-of-band-rows shape (same indexing).
    """
    accel = _rust()
    if accel is None:
        from geometrykit.runtimes._npu import linalg_np

        return linalg_np.banded_cholesky(rows, n, bandwidth)
    from geometrykit import _rust as _ext

    flat = []
    for row in rows:
        pairs = []
        for j, v in row.items():
            pairs.extend((float(j), float(v)))
        flat.append(pairs)
    return _ext.banded_cholesky_grid(flat, int(n), int(bandwidth))


def banded_solve(factor, rhs, n: int, bandwidth: int) -> list:
    """Mirror `linalg_np.banded_solve` under the active engine."""
    accel = _rust()
    if accel is None:
        from geometrykit.runtimes._npu import linalg_np

        return linalg_np.banded_solve(factor, rhs, n, bandwidth)
    from geometrykit import _rust as _ext

    return list(
        _ext.banded_solve_grid(factor, list(rhs), int(n), int(bandwidth))
    )


def eval_point(surf, u: float, v: float):
    """Mirror `surface_np.eval_point` under the active engine."""
    from geometrykit.runtimes._npu import surface_np

    accel = _rust()
    if accel is None:
        return surface_np.eval_point(surf, u, v)
    p = accel.eval_point_grid(surf, [float(u)], [float(v)])
    return surface_np.Vec3(p[0], p[1], p[2])


def eval_tangents(surf, u: float, v: float):
    """Mirror `surface_np.eval_tangents` under the active engine."""
    accel = _rust()
    if accel is None:
        from geometrykit.runtimes._npu import surface_np

        return surface_np.eval_tangents(surf, u, v)
    grid = accel.eval_all_grid(surf, [float(u)], [float(v)])
    return _tangents_from_grid(grid)


def eval_normal(surf, u: float, v: float):
    """Mirror `surface_np.eval_normal` under the active engine.

    Same cross product and normalization expressions as the mirror,
    applied to the dispatched tangents (bit-identical inputs keep the
    result bit-identical).
    """
    su, sv = eval_tangents(surf, u, v)
    nx = su.y * sv.z - su.z * sv.y
    ny = su.z * sv.x - su.x * sv.z
    nz = su.x * sv.y - su.y * sv.x
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    from geometrykit.runtimes._npu import surface_np

    return surface_np.Vec3(nx / n, ny / n, nz / n)


def project_point(surf, pt, m: int = 6, n: int = 6, iters: int = 40):
    """Mirror `surface_np.project_point` under the active engine."""
    accel = _rust()
    if accel is None:
        from geometrykit.runtimes._npu import surface_np

        return surface_np.project_point(surf, pt, m=m, n=n, iters=iters)
    from geometrykit.runtimes._npu import surface_np

    out = accel.project_grid(surf, [pt.x, pt.y, pt.z])
    return (surface_np.Vec3(out[0], out[1], out[2]), (out[3], out[4]))


def _tangents_from_grid(grid: list | tuple):
    from geometrykit.runtimes._npu import surface_np

    return (
        surface_np.Vec3(grid[3], grid[4], grid[5]),
        surface_np.Vec3(grid[6], grid[7], grid[8]),
    )
