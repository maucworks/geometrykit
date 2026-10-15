# This file was generated with the assistance of an AI coding tool.
# Ported from ifckit/ifckit/geometry/curve.py (LGPL-3.0-or-later heritage;
# ifckit is authored by the same copyright holder as geometrykit).
"""
ifckit.geometry.curve
=====================

Curve — a NURBS/BSpline curve with evaluation (De Boor) and IFC serialisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from geometrykit.geometry.primitives import (
    Arc,
    Line,
    Plane,
    Vec,
    _copy_with_fresh_meta,
    _fmt_num,
    _new_id,
    _projected_area,
    _short_id,
)
from geometrykit.geometry.transform import Transform

if TYPE_CHECKING:
    import ifcopenshell

    from geometrykit.geometry.path import Path


# ---------------------------------------------------------------------------
# 4D homogeneous helpers
# ---------------------------------------------------------------------------

_Homo4D = Tuple[float, float, float, float]  # (wx, wy, wz, w)


def _build_full_knots(knots: Sequence[float], multiplicities: Sequence[int]) -> List[float]:
    """Expand compact (knot, mult) pairs into the full knot vector."""
    out: List[float] = []
    for k, m in zip(knots, multiplicities):
        out.extend([float(k)] * m)
    return out


def _cox_de_boor(span: int, p: int, u: float, uknots: List[float]) -> List[float]:
    """Evaluate p+1 non‑zero B‑spline basis functions at *u* (Alg 2.2)."""
    left: List[float] = [0.0] * (p + 1)
    right: List[float] = [0.0] * (p + 1)
    N: List[float] = [0.0] * (p + 1)
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - uknots[span + 1 - j]
        right[j] = uknots[span + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            if abs(denom) > 1e-12:
                t = N[r] / denom
                N[r] = saved + right[r + 1] * t
                saved = left[j - r] * t
            else:
                saved = 0.0
        N[j] = saved
    return N


def _cox_de_boor_deriv(span: int, p: int, u: float, uknots: List[float]) -> List[float]:
    """First derivative of the p+1 non‑zero basis functions at *u* (eq. 3.8).

    Difference formula over the degree-(p−1) row; degenerate (repeated
    clamp) denominators contribute 0 — correct one-sided derivatives at
    clamped edges. Mirrors ``surface_np._basis_and_deriv`` (the Rust port
    carries both).
    """
    low = _cox_de_boor(span, p - 1, u, uknots)
    low_ids = [span - (p - 1) + k for k in range(p)]
    low_map = dict(zip(low_ids, low))
    dN: List[float] = []
    for k in range(p + 1):
        i = span - p + k
        a = uknots[i + p] - uknots[i] if i + p < len(uknots) else 0.0
        b = uknots[i + p + 1] - uknots[i + 1] if i + p + 1 < len(uknots) else 0.0
        t1 = (p * low_map.get(i, 0.0) / a) if abs(a) > 1e-12 else 0.0
        t2 = (p * low_map.get(i + 1, 0.0) / b) if abs(b) > 1e-12 else 0.0
        dN.append(t1 - t2)
    return dN


def _find_knot_span(u: float, p: int, n: int, uknots: List[float]) -> int:
    """Binary search for the knot span containing *u* (Alg 2.1)."""
    if u >= uknots[n + 1]:
        return n
    if u <= uknots[p]:
        return p
    lo, hi = p, n + 1
    mid = (lo + hi) // 2
    while u < uknots[mid] or u >= uknots[mid + 1]:
        if u < uknots[mid]:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) // 2
    return mid


# ---------------------------------------------------------------------------
# Cox‑de Boor evaluation
# ---------------------------------------------------------------------------


def _eval_homo(p: int, uknots: List[float], pts: List[_Homo4D], u: float) -> _Homo4D:
    """Evaluate a (homogeneous) BSpline at *u* via Cox‑de Boor basis functions."""
    n = len(pts) - 1
    span = _find_knot_span(u, p, n, uknots)
    N = _cox_de_boor(span, p, u, uknots)
    res = [0.0, 0.0, 0.0, 0.0]
    for i in range(p + 1):
        idx = span - p + i
        w = pts[idx]
        ni = N[i]
        res[0] += w[0] * ni
        res[1] += w[1] * ni
        res[2] += w[2] * ni
        res[3] += w[3] * ni
    return tuple(res)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Curve
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurveReport:
    """Immutable measure snapshot of a Curve.

    ``length`` is the chord-length approximation (see ``Curve.length``).
    ``area`` is the projected (Newell) area over 50 samples when closed,
    else ``None``.
    """

    type: str = "curve"
    id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    length: float = 0.0
    area: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain (JSON-safe) dict."""
        return {
            "type": self.type,
            "id": self.id,
            "meta": dict(self.meta),
            "length": self.length,
            "area": self.area,
        }

    def __str__(self) -> str:
        return (
            f"CurveReport(id={_short_id(self.id)}, length={_fmt_num(self.length)}, "
            f"area={_fmt_num(self.area)})"
        )


class Curve:
    """A NURBS / BSpline curve evaluated via the De Boor algorithm.

    IFC convention: the parameter lives in the knot domain
    ``[knots[0], knots[-1]]``.  ``Curve.point_at(t)`` accepts a **normalised**
    parameter ``t ∈ [0, 1]`` that is linearly mapped to the knot domain,
    consistent with ``Path.point_at(t)``.

    Args:
        control_points:  Control polygon vertices.
        knots:           Unique knot values (compact form).
        multiplicities:  Multiplicity of each unique knot.
        degree:          Curve degree.
        weights:         Weights for rational NURBS (``None`` = non‑rational).
        closed:          Whether the curve is closed in IFC sense.
    """

    def __init__(
        self,
        control_points: Sequence[Vec],
        knots: Sequence[float],
        multiplicities: Sequence[int],
        degree: int,
        weights: Optional[Sequence[float]] = None,
        closed: bool = False,
        *,
        id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        ncpts = len(control_points)
        nknots = sum(multiplicities)
        expected = ncpts + degree + 1
        if nknots != expected:
            raise ValueError(
                f"Knot vector length ({nknots}) must equal control_points + degree + 1 ({expected})"
            )
        if degree < 1:
            raise ValueError(f"Degree must be >= 1, got {degree}")

        self.degree = degree
        self.points = [Vec(*p) if not isinstance(p, Vec) else p for p in control_points]
        self.knots = list(knots)
        self.multiplicities = list(multiplicities)
        self._weights = list(weights) if weights is not None else None
        self.closed = closed
        self.id = id if id is not None else _new_id()
        self.meta: Dict[str, Any] = dict(meta) if meta else {}

        self._uknots = _build_full_knots(self.knots, self.multiplicities)

    def __copy__(self) -> "Curve":
        """Shallow copy: same id, independent meta dict."""
        return _copy_with_fresh_meta(self)

    # ── properties ──────────────────────────────────────────────────

    @property
    def rational(self) -> bool:
        """Whether the curve or surface is rational (has weights)."""
        return self._weights is not None

    @property
    def knot_domain(self) -> Tuple[float, float]:
        """Knot domain of the curve."""
        return self._uknots[0], self._uknots[-1]

    # ── parameter mapping ──────────────────────────────────────────

    def _u_of_t(self, t: float) -> float:
        u0, u1 = self.knot_domain
        if u1 - u0 < 1e-12:
            return u0
        t = max(0.0, min(1.0, t))
        return u0 + t * (u1 - u0)

    # ── evaluation ──────────────────────────────────────────────────

    def _homo_points(self) -> List[_Homo4D]:
        n = len(self.points)
        w = self._weights if self._weights is not None else [1.0] * n
        return [
            (self.points[i].x * w[i], self.points[i].y * w[i], self.points[i].z * w[i], w[i])
            for i in range(n)
        ]

    def point_at(self, t: float) -> Vec:
        """Evaluate point at normalised parameter ``t ∈ [0, 1]``."""
        u = self._u_of_t(t)
        h = _eval_homo(self.degree, self._uknots, self._homo_points(), u)
        if abs(h[3]) > 1e-12:
            return Vec(h[0] / h[3], h[1] / h[3], h[2] / h[3])
        return Vec(h[0], h[1], h[2])

    def plane_roots(self, origin, normal, samples: int = 200) -> list:
        """Parameters where the curve crosses a plane (no OCC).

        Dense sampling finds sign-change brackets of the signed plane
        distance; each bracket is bisected to machine precision. No
        Newton polish: `tangent_at` is unit, so a Newton step would need
        the parametric speed anyway, and bisection alone already lands
        at 1e-12 -- the polish added risk, not accuracy (measured: unit
        -tangent Newton diverges off tight brackets and clamps to the
        ends). Tangential grazes (no sign change) are not reported --
        stated, not hidden. Returns ascending ``t`` values.
        """
        ox, oy, oz = origin.x, origin.y, origin.z
        nx, ny, nz = normal.x, normal.y, normal.z

        def _signed(t: float) -> float:
            p = self.point_at(t)
            return (p.x - ox) * nx + (p.y - oy) * ny + (p.z - oz) * nz

        values = [(k / samples, _signed(k / samples)) for k in range(samples + 1)]
        roots = []
        for (t0, f0), (t1, f1) in zip(values, values[1:]):
            if (f0 < 0.0) == (f1 < 0.0):
                continue
            a, b, fa = t0, t1, f0
            for _ in range(60):
                m = 0.5 * (a + b)
                fm = _signed(m)
                if (fa < 0.0) == (fm < 0.0):
                    a, fa = m, fm
                else:
                    b = m
                if b - a < 1e-13:
                    break
            t = 0.5 * (a + b)
            if not roots or t - roots[-1] > 1e-9:
                roots.append(t)
        return roots

    def tangent_at(self, t: float) -> Vec:
        """Evaluate unit tangent at normalised parameter ``t ∈ [0, 1]``.

        Analytic derivative (M3, eq. 3.8 with one-sided clamp continuation)
        — matches OCC's `D1` endpoint convention; no finite differences.
        """
        u = self._u_of_t(t)
        from geometrykit.runtimes._npu import curve_np

        Nu, dNu = curve_np._basis_and_deriv(u, self.degree, self._uknots)
        hx = hy = hz = w = 0.0
        dhx = dhy = dhz = dw = 0.0
        for i, ni in Nu.items():
            wi = self._weights[i] if self._weights is not None else 1.0
            p = self.points[i]
            h = ni * wi
            hx += h * p.x
            hy += h * p.y
            hz += h * p.z
            w += h
            dn = dNu.get(i, 0.0) * wi
            dhx += dn * p.x
            dhy += dn * p.y
            dhz += dn * p.z
            dw += dn
        if w == 0.0:
            return Vec(0, 0, 0)
        px, py, pz = hx / w, hy / w, hz / w
        v = Vec(
            dhx - dw * px,
            dhy - dw * py,
            dhz - dw * pz,
        )
        n = v.normalized()
        return n if n is not None else Vec(0, 0, 0)

    def divide(
        self,
        num: Optional[int] = None,
        dist: Optional[float] = None,
        *,
        even: bool = True,
    ) -> List:
        """Distribute points along the curve at equal arc-length intervals.

        Returns a list of ``PathPoint`` objects with attributes ``.t``
        (normalised parameter), ``.point``, and ``.tangent``.

        Exactly one of *num* or *dist* must be given.

        Args:
            num: Number of points (inclusive of start and end).
            dist: Approximate step distance.
                ``even=True`` (default): evenly distributed, spacing ≤ dist,
                endpoints always included.
                ``even=False``: fixed step from start, last step may be shorter.
            even: When ``True`` (default) and *dist* is given, points are
                evenly distributed with spacing ≤ dist.

        Returns:
            List of ``PathPoint`` objects.

        Raises:
            ValueError: If exactly one of *num*/*dist* is not given,
                *num* < 2, *dist* ≤ 0, or the curve has zero length.
        """
        if (num is None) == (dist is None):
            raise ValueError("Exactly one of 'num' or 'dist' must be specified")
        total = self.length
        if total < 1e-12:
            raise ValueError("Cannot divide a zero-length curve")
        if num is not None:
            if num < 2:
                raise ValueError("num must be at least 2")
            step = total / (num - 1)
        else:
            if dist <= 0:
                raise ValueError("dist must be positive")
            if even:
                segs = max(1, math.ceil(total / dist))
                step = total / segs
            else:
                step = dist

        u0, u1 = self.knot_domain
        inv_range = 1.0 / (u1 - u0) if u1 != u0 else 1.0

        try:
            from OCC.Core.GCPnts import GCPnts_AbscissaPoint, GCPnts_UniformAbscissa
            from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
            from OCC.Core.gp import gp_Pnt, gp_Vec

            from geometrykit.geometry.path import PathPoint
            from geometrykit.geometry.surface import _build_occ_curve

            occ = _build_occ_curve(self)
            adaptor = GeomAdaptor_Curve(occ)
            ua = GCPnts_UniformAbscissa(adaptor, step)

            result: list = []
            for i in range(1, ua.NbPoints() + 1):
                u = ua.Parameter(i)
                pnt = gp_Pnt()
                d1 = gp_Vec()
                occ.D1(u, pnt, d1)
                result.append(
                    PathPoint(
                        t=(u - u0) * inv_range,
                        point=Vec(pnt.X(), pnt.Y(), pnt.Z()),
                        tangent=Vec(d1.X(), d1.Y(), d1.Z()).normalized(),
                    )
                )

            if num is not None:
                occ_total = GCPnts_AbscissaPoint.Length(adaptor)
                # OCC's internal arc-length may differ from self.length;
                # ensure exact point count by resampling at correct abscissas
                step_occ = occ_total / (num - 1)
                result2: list = []
                for i in range(num):
                    d = i * step_occ
                    if d > occ_total:
                        d = occ_total
                    ap = GCPnts_AbscissaPoint(1e-6, adaptor, d, u0)
                    u = ap.Parameter() if ap.IsDone() else u0 + i * (u1 - u0) / (num - 1)
                    pnt = gp_Pnt()
                    d1 = gp_Vec()
                    occ.D1(u, pnt, d1)
                    result2.append(
                        PathPoint(
                            t=(u - u0) * inv_range,
                            point=Vec(pnt.X(), pnt.Y(), pnt.Z()),
                            tangent=Vec(d1.X(), d1.Y(), d1.Z()).normalized(),
                        )
                    )
                return result2

            # Remove trailing duplicate if OCC's internal length rounding
            # added an extra point (same t within tolerance)
            while len(result) > 1 and abs(result[-1].t - result[-2].t) < 1e-4:
                result.pop()
            return result

        except ImportError:
            pass

        n = num if num is not None else int(total / step) + 1
        from geometrykit.geometry.path import PathPoint

        return [
            PathPoint(t=t, point=self.point_at(t), tangent=self.tangent_at(t))
            for t in (i / (n - 1) for i in range(n))
        ]

    @property
    def length(self) -> float:
        """Numerical arc length via chord-length approximation."""
        n = max(50, 4 * self.degree)
        pts = self.sample(n)
        return sum((pts[i + 1] - pts[i]).length() for i in range(len(pts) - 1))

    @property
    def area(self) -> Optional[float]:
        """Enclosed area for closed curves, else ``None``.

        Projected (Newell) area over ``sample()`` points — exact for
        polylines, approximate for curved spans. Non-planar loops report
        their projected area.
        """
        if not self.closed:
            return None
        return _projected_area(self.sample())

    def report(self) -> "CurveReport":
        """Immutable measure snapshot (length, area)."""
        return CurveReport(id=self.id, meta=dict(self.meta), length=self.length, area=self.area)

    def sample(self, n: int = 50) -> List[Vec]:
        """Sample *n* evenly‑spaced points in parameter space ``[0, 1]``."""
        return [self.point_at(i / max(n - 1, 1)) for i in range(n)]

    def reverse(self) -> "Curve":
        """Return a new curve with reversed direction.

        The new curve traverses the same geometry from end to start.
        Both control points and knot vector are reflected.
        """
        new_points = list(reversed(self.points))
        new_weights = list(reversed(self._weights)) if self._weights else None

        # Reflect interior knots:  k → k_first + k_last − k
        # then sort ascending to maintain valid knot vector
        k0, k1 = self.knots[0], self.knots[-1]
        interior = sorted(k0 + k1 - k for k in self.knots[1:-1])
        new_knots = [k0] + interior + [k1]

        return Curve(
            control_points=new_points,
            knots=new_knots,
            multiplicities=self.multiplicities,
            degree=self.degree,
            weights=new_weights,
            closed=self.closed,
            id=self.id,
            meta=self.meta,
        )

    # --- affine transforms (pure Python, no OCC needed) ------------------

    def transformed(self, t: "Transform") -> "Curve":
        """Apply a 4×4 affine transform to all control points.

        Works for any affine transform (translation, rotation, scale,
        reflection).  No OCC dependency needed — NURBS control points
        are transformed directly (Piegl & Tiller §6.5).
        """
        return Curve(
            control_points=[t.apply(cp) for cp in self.points],
            knots=list(self.knots),
            multiplicities=list(self.multiplicities),
            degree=self.degree,
            weights=list(self._weights) if self._weights else None,
            closed=self.closed,
            id=self.id,
            meta=self.meta,
        )

    def mirrored(self, plane: "Plane") -> "Curve":
        """Mirror over an arbitrary plane. Returns a new Curve."""
        return self.transformed(Transform.reflection(plane))

    def translated(self, delta: "Vec") -> "Curve":
        """Translate by *delta*. Returns a new Curve."""
        return self.transformed(Transform.translation(delta))

    def rotated(self, axis: "Vec", angle: float) -> "Curve":
        """Rotate around *axis* by *angle* radians. Returns a new Curve."""
        return self.transformed(Transform.rotation(axis, angle))

    def scaled(
        self, sx: float, sy: "Optional[float]" = None, sz: "Optional[float]" = None
    ) -> "Curve":
        """Scale by *sx*, *sy*, *sz*. Returns a new Curve."""
        if sy is None:
            sy = sx
        if sz is None:
            sz = sx
        return self.transformed(Transform.scaling(sx, sy, sz))

    def copy(self) -> "Curve":
        """Return an independent deep copy (same identity)."""
        return Curve(
            control_points=[cp.copy() for cp in self.points],
            knots=list(self.knots),
            multiplicities=list(self.multiplicities),
            degree=self.degree,
            weights=list(self._weights) if self._weights else None,
            closed=self.closed,
            id=self.id,
            meta=self.meta,
        )

    def to_mesh_dict(
        self,
        n_points: int = 50,
        label: str = "",
        material: "dict | None" = None,
        y_up: bool = True,
    ) -> dict:
        """Serialize the curve as a polyline for 3D viewer consumption.

        Args:
            n_points:   Number of sample points.
            label:      Display name.
            material:   Visual properties (color, opacity, …).
            y_up:       If True (default), convert coordinates to
                        Three.js/glTF Y-up: ``(x, z, -y)``.

        Returns:
            A dict with ``primitive``, ``positions``, and optional
            ``label``, ``material`` keys.
        """
        pts = self.sample(n_points)
        if y_up:
            flat = [c for v in pts for c in (v.x, v.z, -v.y)]
        else:
            flat = [c for v in pts for c in (v.x, v.y, v.z)]
        d: dict = {
            "primitive": "line-strip",
            "positions": flat,
            "label": label or "Curve",
        }
        if material is not None:
            d["material"] = material
        return d

    def preview(
        self,
        label: str = "",
        material: "dict | None" = None,
        n_points: int = 50,
        y_up: bool = True,
    ) -> dict:
        """Return a ``__type__: "mesh"`` dict ready for the viewer pipeline."""
        return {
            "__type__": "mesh",
            **self.to_mesh_dict(label=label, material=material, n_points=n_points, y_up=y_up),
        }

    @property
    def start_point(self) -> Vec:
        """Starting point of the path."""
        return self.point_at(0.0)

    @property
    def end_point(self) -> Vec:
        """End point of the path."""
        return self.point_at(1.0)

    # ── bi‑arc fitting ──────────────────────────────────────────────

    def to_biarcs(
        self,
        tol: float = 0.01,
        max_iteration: int = 10,
        min_arc_angle: float = 0.001,
        plane: Optional[Plane] = None,
    ) -> "Path":
        """Approximate this NURBS curve as bi‑arcs → ``Path`` of ``Line`` + ``Arc``.

        Uses recursive bi‑arc fitting (``ifckit.geometry.biarc.fit_biarcs``).

        Args:
            tol:           Maximum deviation from the original curve.
            max_iteration: Maximum recursion depth (default 10).
            min_arc_angle: Arcs with ``|angle| < min_arc_angle`` (rad) are collapsed
                           to lines and G1 is iteratively restored (default 0.001).
            plane:         Optional reference plane.  When given, all points are
                           projected onto this plane and arc normals are set to
                           ``plane.z_axis``.  The resulting ``Path._plane`` is set,
                           enabling downstream methods (``continued``) to use the
                           plane for arc alignment.

        Returns:
            A ``Path`` containing only ``Line`` and ``Arc`` segments.
        """
        from geometrykit.geometry.biarc import fit_biarcs, simplify_biarcs
        from geometrykit.geometry.path import Path

        segments = fit_biarcs(self.point_at, tolerance=tol, max_depth=max_iteration)
        if min_arc_angle > 0:
            segments = simplify_biarcs(segments, min_angle=min_arc_angle)
        path = Path(plane=plane)
        for seg in segments:
            if isinstance(seg, Arc):
                if plane is not None:
                    center_p = plane.closest_point(seg.center)
                    start_p = plane.closest_point(seg.start)
                    sign = 1.0 if (seg.normal @ plane.z_axis) >= 0 else -1.0
                    seg = Arc(center_p, plane.z_axis, start_p, seg.angle * sign)
                path._segments.append(seg)
            elif isinstance(seg, Line):
                if plane is not None:
                    seg = Line(plane.closest_point(seg.start), plane.closest_point(seg.end))
                path._segments.append(seg)
        return path

    def to_path(
        self,
        tolerance: float = 0.01,
        max_depth: int = 10,
        min_arc_angle: float = 0.001,
    ) -> "Path":
        """Alias for :meth:`to_biarcs`."""
        return self.to_biarcs(tol=tolerance, max_iteration=max_depth, min_arc_angle=min_arc_angle)

    # ── IFC serialisation ──────────────────────────────────────────

    def _to_ifc_points(self, ifc_file: "ifcopenshell.file") -> list:
        from geometrykit.geometry._bridge import require_bridge

        return [require_bridge().pt3(ifc_file, v.x, v.y, v.z) for v in self.points]

    def to_ifc_bspline(self, ifc_file: "ifcopenshell.file") -> "ifcopenshell.entity_instance":
        """Create an ``IfcBSplineCurveWithKnots`` (non‑rational)."""
        if self.rational:
            raise ValueError("Use to_ifc_rational() for rational NURBS curves")
        e = ifc_file.create_entity("IfcBSplineCurveWithKnots")
        e.Degree = self.degree
        e.ControlPointsList = self._to_ifc_points(ifc_file)
        e.CurveForm = "UNSPECIFIED"
        e.ClosedCurve = self.closed
        e.SelfIntersect = False
        e.Knots = self.knots
        e.KnotMultiplicities = self.multiplicities
        e.KnotSpec = "UNSPECIFIED"
        return e

    def to_ifc_rational(self, ifc_file: "ifcopenshell.file") -> "ifcopenshell.entity_instance":
        """Create an ``IfcRationalBSplineCurveWithKnots``."""
        if not self.rational:
            raise ValueError("Use to_ifc_bspline() for non‑rational curves")
        e = ifc_file.create_entity("IfcRationalBSplineCurveWithKnots")
        e.Degree = self.degree
        e.ControlPointsList = self._to_ifc_points(ifc_file)
        e.CurveForm = "UNSPECIFIED"
        e.ClosedCurve = self.closed
        e.SelfIntersect = False
        e.Knots = self.knots
        e.KnotMultiplicities = self.multiplicities
        e.KnotSpec = "UNSPECIFIED"
        e.WeightsData = self._weights
        return e

    # ── serialisation ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (incl. identity)."""
        d: Dict[str, Any] = {
            "degree": self.degree,
            "control_points": [v.to_dict() for v in self.points],
            "knots": self.knots,
            "multiplicities": self.multiplicities,
            "closed": self.closed,
            "id": self.id,
            "meta": dict(self.meta),
        }
        if self._weights is not None:
            d["weights"] = self._weights
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Curve":
        """Deserialize from a dict (missing id mints a new one)."""
        return cls(
            control_points=[Vec.from_dict(v) for v in d["control_points"]],
            knots=d["knots"],
            multiplicities=d["multiplicities"],
            degree=d["degree"],
            weights=d.get("weights"),
            closed=d.get("closed", False),
            id=d.get("id"),
            meta=d.get("meta"),
        )

    def __repr__(self) -> str:
        w_str = ", rational" if self.rational else ""
        return (
            f"Curve(degree={self.degree}, "
            f"{len(self.points)} points, "
            f"knots={self.knots[:3]}...{self.knots[-3:]}{w_str})"
        )

    # ── Hermite-style construction ──────────────────────────────────

    @classmethod
    def from_tangents(
        cls,
        start: Vec,
        tan_start: Vec,
        end: Vec,
        tan_end: Vec,
        scale: float | None = None,
    ) -> "Curve":
        """Create a cubic Bezier curve from two points and their tangents.

        The curve passes through *start* and *end* with the given
        tangent directions at each endpoint.  The tangent **magnitude**
        determines the curve stiffness — a longer tangent pulls the
        curve further before bending toward the other endpoint.

        Args:
            start:      Start point.
            tan_start:  Start tangent (direction + magnitude).
            end:        End point.
            tan_end:    End tangent (direction + magnitude).
            scale:      Optional uniform multiplier to override both
                        tangent magnitudes.  When ``None`` (default)
                        the magnitude of each tangent vector is used
                        directly (falling back to ``|end - start| / 3``
                        for zero‑length tangents).

        Returns:
            A new degree‑3 BSpline (cubic Bezier) Curve.
        """
        chord = end - start
        default = chord.length() / 3.0

        ts_len = tan_start.length()
        te_len = tan_end.length()
        use_start = ts_len if ts_len > 1e-12 else default
        use_end = te_len if te_len > 1e-12 else default

        if scale is not None:
            use_start = use_end = scale

        ts = tan_start.normalized() * use_start if ts_len > 1e-12 else Vec(0, 0, 0)
        te = tan_end.normalized() * use_end if te_len > 1e-12 else Vec(0, 0, 0)

        control_points = [start, start + ts, end - te, end]
        knots = [0.0, 1.0]
        mults = [4, 4]
        return cls(
            control_points=control_points,
            knots=knots,
            multiplicities=mults,
            degree=3,
        )

    # ── Interpolation through points ────────────────────────────────

    @classmethod
    def from_points(
        cls,
        points: "Sequence[Vec]",
        degree: int = 3,
        tan_start: "Vec | None" = None,
        tan_end: "Vec | None" = None,
        knots: "Sequence[float] | None" = None,
    ) -> "Curve":
        """Interpolate a BSpline curve through a set of points.

        Pure kernel (NURBS Book A9.1 / A9.2 end conditions) — no pythonocc
        needed. The OCC oracle path lives in the test suite.

        Args:
            points:    Points the curve must pass through (minimum 2).
            degree:    Target degree (default 3).  Minimum is clamped to
                      ``min(len(points) - 1, 3)``.
            tan_start: Optional start tangent direction for G1 end
                      condition. The derivative magnitude is the natural
                      chord one (unscaled convention).
            tan_end:   Optional end tangent direction for G1 end condition.
            knots:     Optional explicit knot parameters ``∈ [0, 1]``.
                      Length must equal ``len(points)``.  When ``None``
                      chord‑length parameterisation is used.

        Returns:
            A new ``Curve`` through all *points*.
        """
        from geometrykit.runtimes._npu import curve_np

        if len(points) < 2:
            raise ValueError("Need at least 2 points")
        degree_clamped = max(1, min(degree, len(points) - 1))
        cpts, knots, mults = curve_np.interpolate(
            points, degree=degree_clamped,
            tan_start=tan_start, tan_end=tan_end, knots_param=knots,
        )
        return cls(
            control_points=[Vec(*c) for c in cpts],
            knots=knots,
            multiplicities=mults,
            degree=degree_clamped,
            closed=False,
        )

    @classmethod
    def assemble(
        cls,
        curves: "Sequence[Curve]",
        tol: float = 1e-6,
    ) -> "List[Curve]":
        """Join end-to-start connected curves into continuous BSpline curves.

        Pure kernel: C0 knot-vector assembly via ``curve_np.join_two``
        (see the plan's M3).  Curves that are not G0-connected (gap larger
        than *tol*) split into separate groups; orientation is repaired
        (4-way endpoint matching).  Joined curves within a group must
        share one degree; groups with mixed degrees stay split.

        Args:
            curves:  Sequence of curves to join.
            tol:     G0 tolerance for joining.

        Returns:
            List of joined ``Curve`` objects.
        """
        from geometrykit.runtimes._npu import curve_np

        if not curves:
            return []

        groups = curve_np.group_and_orient(curves, tol=tol)
        result: list = []
        for grp in groups:
            joined = grp[0]
            ok = True
            for nxt in grp[1:]:
                if joined.rational or nxt.rational:
                    ok = False
                    break
                merged = curve_np.join_two(joined, nxt, tol=tol)
                if merged is None:
                    ok = False
                    break
                cpts, knots, mults = merged
                joined = cls(
                    control_points=cpts,
                    knots=knots,
                    multiplicities=mults,
                    degree=joined.degree,
                    closed=False,
                )
            if ok:
                result.append(joined)
                continue
            # degree mismatch (or rationality): emit the group as-is
            result.extend(grp)

        return result

    @classmethod
    def from_occ_edge(cls, edge) -> "Curve":
        """Create a Curve from an OCC ``TopoDS_Edge``.

        Supports both FreeCAD ``Part.Edge`` and raw ``pythonocc-core``
        ``TopoDS_Edge``.  The edge must contain a ``Geom_BSplineCurve``
        — lines and circles are not supported (use ``Path`` for those).
        """
        # Try FreeCAD API (edge.Curve.getPoles)
        curve_attr = getattr(edge, "Curve", None)
        if curve_attr is not None and hasattr(curve_attr, "getPoles"):
            curve = curve_attr
            poles = [Vec(p.x, p.y, p.z) for p in curve.getPoles()]
            knots = list(curve.getKnots())
            mults = list(curve.getMultiplicities())
            degree = curve.Degree
            weights = list(curve.getWeights()) if curve.isRational() else None
            closed = curve.isPeriodic()
            return cls(
                control_points=poles,
                knots=knots,
                multiplicities=mults,
                degree=degree,
                weights=weights,
                closed=closed,
            )

        # Fall back to pythonocc-core API
        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.Geom import Geom_BSplineCurve
        except ImportError:
            raise ImportError("Curve.from_occ_edge() requires either FreeCAD or pythonocc-core")

        result = BRep_Tool.Curve(edge)
        if result is None:
            raise TypeError("Edge has no curve")
        curve_handle, first_param, last_param = result

        # Downcast to BSpline
        bspline = Geom_BSplineCurve.DownCast(curve_handle)
        if bspline is None:
            raise TypeError("Edge contains a non-BSpline curve. Use Path for line/circle segments.")

        # If the edge is trimmed (parameter range ≠ full curve domain),
        # sample the edge within its actual bounds — the raw BSpline
        # extends beyond the edge's trim limits.
        k0 = bspline.Knot(1)
        k1 = bspline.Knot(bspline.NbKnots())
        eps = 1e-4 * max(1.0, abs(k1 - k0))
        if abs(first_param - k0) > eps or abs(last_param - k1) > eps:
            from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

            adaptor = BRepAdaptor_Curve(edge)
            n = max(20, bspline.Degree() * 4)
            pts = []
            for i in range(n):
                u = first_param + (last_param - first_param) * i / (n - 1)
                p = adaptor.Value(u)
                pts.append(Vec(p.X(), p.Y(), p.Z()))
            try:
                return cls.from_points(pts, degree=min(3, len(pts) - 1))
            except Exception:
                pass  # fall through to full-curve extraction below

        poles = []
        for i in range(bspline.NbPoles()):
            p = bspline.Pole(i + 1)
            poles.append(Vec(p.X(), p.Y(), p.Z()))

        knots = [bspline.Knot(i + 1) for i in range(bspline.NbKnots())]
        mults = [bspline.Multiplicity(i + 1) for i in range(bspline.NbKnots())]
        degree = bspline.Degree()
        weights = (
            [bspline.Weight(i + 1) for i in range(bspline.NbPoles())]
            if bspline.IsRational()
            else None
        )
        closed = bspline.IsPeriodic()

        return cls(
            control_points=poles,
            knots=knots,
            multiplicities=mults,
            degree=degree,
            weights=weights,
            closed=closed,
        )
