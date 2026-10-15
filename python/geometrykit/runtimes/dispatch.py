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
