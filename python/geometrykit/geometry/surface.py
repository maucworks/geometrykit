# This file was generated with the assistance of an AI coding tool.
# Ported from ifckit/ifckit/geometry/surface.py (LGPL-3.0-or-later heritage;
# ifckit is authored by the same copyright holder as geometrykit).
"""
ifckit.geometry.surface
=======================

Surface — a NURBS / BSpline surface with IFC serialisation and optional
OCC (``pythonocc-core``) evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from geometrykit.geometry.primitives import Vec, _copy_with_fresh_meta, _fmt_num, _new_id, _short_id
from geometrykit.geometry.transform import Transform

if TYPE_CHECKING:
    import ifcopenshell
    from ifckit.schema import TessellationDetail  # noqa: F401

    from geometrykit.geometry.curve import Curve  # noqa: F401
    from geometrykit.geometry.primitives import Plane  # noqa: F401

# ---------------------------------------------------------------------------
# Optional OCC detection
# ---------------------------------------------------------------------------

_HAS_OCC = False
try:
    from OCC.Core.Geom import Geom_BSplineSurface
    from OCC.Core.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Vec
    from OCC.Core.TColgp import TColgp_Array2OfPnt
    from OCC.Core.TColStd import (
        TColStd_Array1OfInteger,
        TColStd_Array1OfReal,
        TColStd_Array2OfReal,
    )

    _HAS_OCC = True
except ImportError:
    pass


def require_occ():
    """Import and return the OCC module, or raise ImportError."""
    if not _HAS_OCC:
        raise ImportError(
            "pythonocc-core is required. It has no PyPI distribution — "
            "install from conda-forge: conda install -c conda-forge pythonocc-core"
        )


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceReport:
    """Immutable measure snapshot of a Surface.

    ``area`` sums ``occ_tessellate`` triangles (deflection 0.01) and is
    ``None`` without ``pythonocc-core`` — consistent with
    ``Surface.to_mesh_dict``, which also requires OCC.
    """

    type: str = "surface"
    id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    area: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain (JSON-safe) dict."""
        return {
            "type": self.type,
            "id": self.id,
            "meta": dict(self.meta),
            "area": self.area,
        }

    def __str__(self) -> str:
        return f"SurfaceReport(id={_short_id(self.id)}, area={_fmt_num(self.area)})"


class Surface:
    """A NURBS / BSpline surface (tensor product).

    Data + IFC serialisation in pure Python.  Evaluation and
    surface‑plane intersection require ``pythonocc-core`` (optional).
    """

    def __init__(
        self,
        control_points: Sequence[Sequence[Vec]],
        uknots: Sequence[float],
        vknots: Sequence[float],
        umults: Sequence[int],
        vmults: Sequence[int],
        udegree: int,
        vdegree: int,
        weights: Optional[Sequence[Sequence[float]]] = None,
        uclosed: bool = False,
        vclosed: bool = False,
        *,
        id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.control_points = [
            [Vec(*p) if not isinstance(p, Vec) else p for p in row] for row in control_points
        ]
        self.uknots = [float(k) for k in uknots]
        self.vknots = [float(k) for k in vknots]
        self.umults = [int(m) for m in umults]
        self.vmults = [int(m) for m in vmults]
        self.udegree = udegree
        self.vdegree = vdegree
        self._weights = [list(w) for w in weights] if weights is not None else None
        self.uclosed = uclosed
        self.vclosed = vclosed
        self.id = id if id is not None else _new_id()
        self.meta: Dict[str, Any] = dict(meta) if meta else {}
        self._occ_face = None  # cached OCC TopoDS_Face for optimised MakeFilling
        self._occ_edge = None  # cached OCC TopoDS_Edge matching curve in _occ_face

    def __copy__(self) -> "Surface":
        """Shallow copy: same id, independent meta dict."""
        return _copy_with_fresh_meta(self)

    # ── properties ──────────────────────────────────────────────────

    @property
    def rational(self) -> bool:
        """Whether the curve or surface is rational (has weights)."""
        return self._weights is not None

    @property
    def nu(self) -> int:
        """Number of control points in the U direction."""
        return len(self.control_points)

    @property
    def nv(self) -> int:
        """Number of control points in the V direction."""
        return len(self.control_points[0]) if self.control_points else 0

    @property
    def area(self) -> Optional[float]:
        """Surface area via OCC tessellation, else ``None`` without OCC.

        Sums ``occ_tessellate`` triangles (deflection 0.01). Requires
        ``pythonocc-core`` — returns ``None`` when it is missing instead
        of raising, so reports stay usable in OCC-less environments.
        """
        try:
            verts, tris = occ_tessellate(self)
        except ImportError:
            # require_occ() inside occ_tessellate: no pythonocc-core.
            return None
        total = 0.0
        for a, b, c in tris:
            pa, pb, pc = (Vec(*verts[i - 1]) for i in (a, b, c))
            total += abs((pb - pa) ** (pc - pa)) * 0.5
        return total

    def report(self) -> "SurfaceReport":
        """Immutable measure snapshot (area)."""
        return SurfaceReport(id=self.id, meta=dict(self.meta), area=self.area)

    # ── evaluation (via OCC) ───────────────────────────────────────

    def point_at(self, u: float, v: float) -> Vec:
        """Evaluate surface point at ``(u, v)`` — both in ``[0, 1]``.

        Pure kernel path (``runtimes/_npu``, M2): no pythonocc needed.
        """
        from geometrykit.runtimes._npu import surface_np

        p = surface_np.eval_point(self, u, v)
        return Vec(p.x, p.y, p.z)

    def tangents_at(self, u: float, v: float) -> "tuple[Vec, Vec]":
        """Analytic first-order derivatives ``(dS/du, dS/dv)`` at ``(u, v)``."""
        from geometrykit.runtimes._npu import surface_np

        su, sv = surface_np.eval_tangents(self, u, v)
        return Vec(su.x, su.y, su.z), Vec(sv.x, sv.y, sv.z)

    def normal_at(self, u: float, v: float) -> Vec:
        """Unit normal at ``(u, v)`` (cross of the analytic tangents; kernel path)."""
        from geometrykit.runtimes._npu import surface_np

        n = surface_np.eval_normal(self, u, v)
        return Vec(n.x, n.y, n.z)

    def project(
        self, pt: Vec, m: int = 6, n: int = 6, iters: int = 40
    ) -> "tuple[Vec, tuple[float, float]]":
        """Closest surface point to ``pt``: ``(proj, (u, v))`` (kernel Newton).

        ``occ_project_point`` remains available as the OCCT oracle for tests.
        """
        from geometrykit.runtimes._npu import surface_np

        p, uv = surface_np.project_point(self, pt, m=m, n=n, iters=iters)
        return Vec(p.x, p.y, p.z), uv

    def to_mesh_dict(
        self,
        nu: int = 20,
        nv: int = 20,
        label: str = "",
        material: "dict | None" = None,
        deflection: "float | object" = 0.01,
        y_up: bool = True,
    ) -> dict:
        """Triangulate the surface via OCC and return a dict for 3D viewer.

        Uses OCC ``BRepMesh_IncrementalMesh`` (requires ``pythonocc-core``).

        Args:
            nu:         Ignored (OCC adaptive mesh).
            nv:         Ignored (OCC adaptive mesh).
            label:      Display name.
            material:   Visual properties (color, opacity, …).
            deflection: Mesh deflection / ``TessellationDetail``.
            y_up:       If True (default), convert coordinates to
                        Three.js/glTF Y-up: ``(x, z, -y)``.

        Returns:
            A dict with ``primitive="triangles"``, ``positions``,
            ``indices``, and optional ``label``, ``material``.
        """
        from geometrykit.geometry.surface import occ_tessellate, require_occ

        require_occ()
        verts, tris = occ_tessellate(self, deflection)
        if y_up:
            positions = [c for v in verts for c in (v[0], v[2], -v[1])]
        else:
            positions = [c for v in verts for c in v]
        indices = [i - 1 for tri in tris for i in tri]

        d: dict = {
            "primitive": "triangles",
            "positions": positions,
            "indices": indices,
            "label": label or "Surface",
        }
        if material is not None:
            d["material"] = material
        return d

    def preview(
        self,
        label: str = "",
        material: "dict | None" = None,
        deflection: "float | object" = 0.01,
        y_up: bool = True,
    ) -> dict:
        """Return a ``__type__: "mesh"`` dict ready for the viewer pipeline.

        Uses OCC adaptive triangulation. See ``Path.preview()`` for
        the full material documentation.

        Args:
            label:      Display name.
            material:   Visual properties dict.
            deflection: Mesh deflection / ``TessellationDetail`` (see ifckit).
            y_up:       If True (default), convert coordinates to
                        Three.js/glTF Y-up: ``(x, z, -y)``.

        Returns:
            ``{"__type__": "mesh", "primitive": "triangles", …}``
        """
        return {
            "__type__": "mesh",
            **self.to_mesh_dict(label=label, material=material, deflection=deflection, y_up=y_up),
        }

    # ── IFC serialisation ──────────────────────────────────────────

    def _to_ifc_pt(self, ifc_file, v: Vec):
        return ifc_file.create_entity("IfcCartesianPoint", Coordinates=(v.x, v.y, v.z))

    def _ifc_points(self, ifc_file) -> List[List]:
        return [[self._to_ifc_pt(ifc_file, p) for p in row] for row in self.control_points]

    def _set_common(self, e, ifc_file) -> None:
        e.UDegree = self.udegree
        e.VDegree = self.vdegree
        e.ControlPointsList = self._ifc_points(ifc_file)
        e.SurfaceForm = "UNSPECIFIED"
        e.UClosed = self.uclosed
        e.VClosed = self.vclosed
        e.SelfIntersect = False
        e.UMultiplicities = self.umults
        e.VMultiplicities = self.vmults
        e.UKnots = self.uknots
        e.VKnots = self.vknots
        e.KnotSpec = "UNSPECIFIED"

    def to_ifc_bspline(self, ifc_file) -> "ifcopenshell.entity_instance":
        """Convert to an IFC B-spline surface representation."""
        if self.rational:
            raise ValueError("Use to_ifc_rational() for rational surfaces")
        e = ifc_file.create_entity("IfcBSplineSurfaceWithKnots")
        self._set_common(e, ifc_file)
        return e

    def to_ifc_rational(self, ifc_file) -> "ifcopenshell.entity_instance":
        """Convert to an IFC rational B-spline surface representation."""
        if not self.rational:
            raise ValueError("Use to_ifc_bspline() for non‑rational surfaces")
        e = ifc_file.create_entity("IfcRationalBSplineSurfaceWithKnots")
        self._set_common(e, ifc_file)
        e.WeightsData = self._weights
        return e

    # ── dict serialisation ─────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (incl. identity)."""
        d: Dict[str, Any] = {
            "udegree": self.udegree,
            "vdegree": self.vdegree,
            "control_points": [[v.to_dict() for v in row] for row in self.control_points],
            "uknots": self.uknots,
            "vknots": self.vknots,
            "umults": self.umults,
            "vmults": self.vmults,
            "uclosed": self.uclosed,
            "vclosed": self.vclosed,
            "id": self.id,
            "meta": dict(self.meta),
        }
        if self._weights is not None:
            d["weights"] = self._weights
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Surface":
        """Deserialize from a dict (missing id mints a new one)."""
        return cls(
            control_points=[[Vec.from_dict(v) for v in row] for row in d["control_points"]],
            uknots=d["uknots"],
            vknots=d["vknots"],
            umults=d["umults"],
            vmults=d["vmults"],
            udegree=d["udegree"],
            vdegree=d["vdegree"],
            weights=d.get("weights"),
            uclosed=d.get("uclosed", False),
            vclosed=d.get("vclosed", False),
            id=d.get("id"),
            meta=d.get("meta"),
        )

    def __repr__(self) -> str:
        w_str = ", rational" if self.rational else ""
        return f"Surface({self.nu}×{self.nv} pts, U{self.udegree}/V{self.vdegree}{w_str})"

    # --- affine transforms (pure Python, no OCC needed) ------------------

    def transformed(self, t: "Transform") -> "Surface":
        """Apply a 4×4 affine transform to all control points.

        Works for any affine transform. Pure Python — no OCC dependency.
        """
        weights = None
        if self._weights is not None:
            weights = [list(w) for w in self._weights]
        return Surface(
            control_points=[[t.apply(cp) for cp in row] for row in self.control_points],
            uknots=list(self.uknots),
            vknots=list(self.vknots),
            umults=list(self.umults),
            vmults=list(self.vmults),
            udegree=self.udegree,
            vdegree=self.vdegree,
            weights=weights,
            uclosed=self.uclosed,
            vclosed=self.vclosed,
            id=self.id,
            meta=self.meta,
        )

    def mirrored(self, plane: "Plane") -> "Surface":
        """Mirror over an arbitrary plane. Returns a new Surface."""
        return self.transformed(Transform.reflection(plane))

    def translated(self, delta: "Vec") -> "Surface":
        """Translate by *delta*. Returns a new Surface."""
        return self.transformed(Transform.translation(delta))

    def rotated(self, axis: "Vec", angle: float) -> "Surface":
        """Rotate around *axis* by *angle* radians. Returns a new Surface."""
        return self.transformed(Transform.rotation(axis, angle))

    def scaled(
        self, sx: float, sy: "Optional[float]" = None, sz: "Optional[float]" = None
    ) -> "Surface":
        """Scale by *sx*, *sy*, *sz*. Returns a new Surface."""
        if sy is None:
            sy = sx
        if sz is None:
            sz = sx
        return self.transformed(Transform.scaling(sx, sy, sz))

    def copy(self) -> "Surface":
        """Return an independent deep copy (same identity)."""
        weights = None
        if self._weights is not None:
            weights = [[w for w in row] for row in self._weights]
        return Surface(
            control_points=[[cp.copy() for cp in row] for row in self.control_points],
            uknots=list(self.uknots),
            vknots=list(self.vknots),
            umults=list(self.umults),
            vmults=list(self.vmults),
            udegree=self.udegree,
            vdegree=self.vdegree,
            weights=weights,
            uclosed=self.uclosed,
            vclosed=self.vclosed,
            id=self.id,
            meta=self.meta,
        )

    # ── OCC bridge ─────────────────────────────────────────────────

    @classmethod
    def from_occ_surface(cls, occ_surface) -> "Surface":
        """Create a Surface from an OCC surface object."""
        require_occ()
        udeg = occ_surface.UDegree()
        vdeg = occ_surface.VDegree()
        nu = occ_surface.NbUPoles()
        nv = occ_surface.NbVPoles()

        poles = []
        for i in range(nu):
            row = []
            for j in range(nv):
                p = occ_surface.Pole(i + 1, j + 1)
                row.append(Vec(p.X(), p.Y(), p.Z()))
            poles.append(row)

        uknots = [occ_surface.UKnot(i + 1) for i in range(occ_surface.NbUKnots())]
        vknots = [occ_surface.VKnot(i + 1) for i in range(occ_surface.NbVKnots())]
        umults = [occ_surface.UMultiplicity(i + 1) for i in range(occ_surface.NbUKnots())]
        vmults = [occ_surface.VMultiplicity(i + 1) for i in range(occ_surface.NbVKnots())]

        rational = occ_surface.IsURational() or occ_surface.IsVRational()
        weights = None
        if rational:
            weights = [[occ_surface.Weight(i + 1, j + 1) for j in range(nv)] for i in range(nu)]

        return cls(
            control_points=poles,
            uknots=uknots,
            vknots=vknots,
            umults=umults,
            vmults=vmults,
            udegree=udeg,
            vdegree=vdeg,
            weights=weights,
            uclosed=occ_surface.IsUPeriodic(),
            vclosed=occ_surface.IsVPeriodic(),
        )

    # ── loft & fill ────────────────────────────────────────────────

    @classmethod
    def loft(cls, curves: "Sequence[Curve]", degree: int = 3) -> "Surface":
        """Create a surface through a series of curves (loft).

        Args:
            curves:  Cross‑section curves (ordered).
            degree:  Target degree in the loft direction.

        Returns:
            A new Surface.
        """
        require_occ()
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopExp import TopExp_Explorer

        lofter = BRepOffsetAPI_ThruSections(False)
        for cv in curves:
            edge = _build_occ_edge(cv)
            wire = BRepBuilderAPI_MakeWire(edge).Wire()
            lofter.AddWire(wire)
        lofter.Build()
        if not lofter.IsDone():
            raise RuntimeError("Loft failed — ThruSections returned IsDone=False")

        exp = TopExp_Explorer(lofter.Shape(), TopAbs_FACE)
        if not exp.More():
            raise RuntimeError("Loft produced no faces")
        face = exp.Current()
        adaptor = BRepAdaptor_Surface(face)
        geom = adaptor.Surface().Surface()
        bspline = Geom_BSplineSurface.DownCast(geom)
        if bspline is None:
            raise RuntimeError("Loft result is not a BSpline surface")

        return cls.from_occ_surface(bspline)

    @classmethod
    def fill(cls, curves, degree=3, n_samples=10):
        """Create a surface bounded by 3‑4 curves via Coons blending.

        The interior is estimated via bilinear Coons blending from the
        boundary curves, then a BSpline surface is fit through the
        resulting point grid.  Edges are auto‑aligned to the Coons
        convention (u‑edges left→right, v‑edges bottom→top).

        For per‑edge G1 constraints and support surfaces use
        :meth:`patch`.

        Args:
            curves:    Boundary curves (3 or 4).
            degree:    Target degree.
            n_samples: Number of subdivisions in U and V.

        Returns:
            A new Surface.
        """
        n = len(curves)
        if n not in (3, 4):
            raise ValueError("fill() requires 3 or 4 boundary curves. For other sizes use patch().")

        # Auto‑align to Coons convention
        curves = list(curves)
        eps = 1e-4
        for i in range(n):
            s = curves[i].point_at(0)
            e = curves[i].point_at(1)
            if i in (0, 2):  # u‑direction edge (bottom / top)
                if s.x > e.x + eps:
                    curves[i] = curves[i].reverse()
            elif i in (1, 3):  # v‑direction edge (right / left)
                if s.y > e.y + eps:
                    curves[i] = curves[i].reverse()

        require_occ()
        from OCC.Core.GeomAbs import GeomAbs_C2
        from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.TColgp import TColgp_Array2OfPnt

        ns = n_samples
        c0, c1, c2 = curves[0], curves[1], curves[2]
        c3 = curves[3] if n == 4 else None

        grid = TColgp_Array2OfPnt(1, ns + 1, 1, ns + 1)
        for i in range(ns + 1):
            u = i / ns
            for j in range(ns + 1):
                v = j / ns
                pu = c0.point_at(u) * (1.0 - v) + c2.point_at(u) * v
                pv = (c3 or c1).point_at(v) * (1.0 - u) + c1.point_at(v) * u
                corners = (
                    c0.point_at(0) * (1.0 - u) * (1.0 - v)
                    + c0.point_at(1) * u * (1.0 - v)
                    + (c2.point_at(1) if n == 4 else c0.point_at(1)) * u * v
                    + (c2.point_at(0) if n == 4 else c0.point_at(0)) * (1.0 - u) * v
                )
                p = pu + pv - corners
                grid.SetValue(i + 1, j + 1, gp_Pnt(p.x, p.y, p.z))

        dg = degree
        if dg > ns:
            dg = ns
        occ_surf = GeomAPI_PointsToBSplineSurface(
            grid,
            3,
            max(3, dg + 2),
            GeomAbs_C2,
            1e-3,
        )
        return cls.from_occ_surface(occ_surf.Surface())

    @classmethod
    def patch(
        cls,
        curves,
        constraints=None,
        supports=None,
        tol: float = 1e-3,
        tolerance: "float | None" = None,
        fill: "str | None" = None,
        tangency: "float | None" = None,
        g2: bool = False,
        nu: int = 8,
        nv: int = 8,
        parametrization: str = "arclength",
        adaptive: float = 0.0,
        loose: int = 0,
        corner_stations: int = 1,
    ) -> "Surface":
        """Create a surface bounded by ≥2 curves, optionally G1-constrained.

        Supports per‑edge **G1** (tangent) continuity against optional
        support surfaces.  Edges are auto-aligned to form a closed loop.

        Args:
            curves:      Boundary curves (≥2).
            constraints: Per‑edge continuity ``["C0", "G1", …]``.
                        ``None`` or ``"C0"`` → position only.
            supports:    Per‑edge support ``Surface`` for G1 edges
                        (``None`` = no support). G1 edges require a
                        support.
            tol:         3D tolerance for MakeFilling.  Higher values
                        (e.g. ``1e-2``) are faster but coarser.
            tolerance:   Deprecated alias for *tol*.
            fill:        Which solver builds a four-curve patch: ``None``
                        (legacy -- the OCCT twin for G1 edges, the P-0
                        Coons blend for C0), ``"coons"`` (the P-0 blend,
                        explicit), or ``"energy"`` (the energy-minimising
                        plate fill with the continuity conditions as
                        boundary conditions of the solve, rows-then-fit
                        when adaptive is set). ``"energy"`` needs exactly
                        four curves and is OCC-free.
            tangency:    With ``fill="energy"``: the boundary
                        cross-derivative's magnitude as a multiple of the
                        support's own slope (``None`` = direction only).
            g2:          With ``fill="energy"``: also match the support's
                        cross-curvature, as a second pass over the G1
                        result.
            nu/nv:       Station counts for the fill grid.
            parametrization: how the fill stations are spread --
                        ``"arclength"`` (even in space, uniform knots) or
                        ``"chord"`` (the curves' own parameterisation).
            adaptive:    Tolerance for the adaptive re-fit (0 = off).
            loose:       Stations of enlargement per side for the loose
                        fill (0 = the classical fill).
            corner_stations: stations dropped at each end of an edge.

        Returns:
            A new Surface.

        Raises:
            ValueError: If G1 is specified without a support surface.
        """
        if tolerance is not None:
            tol = tolerance
        n = len(curves)
        if n < 2:
            raise ValueError("patch() requires at least 2 curves")

        # Route: the energy fill (OCC-free kernel pipeline), the pure
        # position-only P-0 blend, or the legacy OCCT twin.
        constraints = list(constraints) if constraints is not None else ["C0"] * n
        _needs_g1 = any(
            (c or "C0").upper() == "G1" for c in constraints
        )
        if fill is not None and fill not in ("coons", "energy"):
            raise ValueError(
                f'unknown fill {fill!r}: use None (legacy), "coons" or "energy"'
            )
        if fill in ("coons", "energy"):
            # Kernel path (OCC-free): the base is the Coons blend or the
            # energy plate; G1/G2 rows apply on top when asked.
            if n != 4:
                raise NotImplementedError(
                    f'fill="{fill}" supports exactly four boundary curves for now '
                    "(n-sided chart machinery is a later increment)"
                )
            from geometrykit.runtimes._npu import patch_g1, patch_np

            params_u, params_v, oriented = patch_np.coons_parametrization(
                curves, nu=nu, nv=nv, tol=1e-6, parametrization=parametrization
            )
            towards = Vec(
                sum(c.point_at(0.5).x for c in curves) / n,
                sum(c.point_at(0.5).y for c in curves) / n,
                sum(c.point_at(0.5).z for c in curves) / n,
            )
            fields = {}
            for edge, spine, support, want in zip(
                patch_g1.EDGE_NAMES, curves, supports or [None] * n, constraints
            ):
                if (want or "C0").upper() != "G1":
                    continue
                if support is None:
                    raise ValueError(
                        f"G1 constraint on {edge} requires a support Surface "
                        "(the adjacent face that defines the tangent direction)."
                    )
                fields[edge] = patch_g1.spine_field(
                    edge, spine, support, oriented, params_u, params_v, towards
                )
            surface, _corners, _stations = patch_g1.constrained_patch(
                curves, fields, nu=nu, nv=nv, tol=1e-6, degree=3,
                tangency=tangency, g2=g2, fill=(fill == "energy"),
                loose=loose, corner_stations=corner_stations,
                adaptive=adaptive, parametrization=parametrization,
            )
            return surface
        if not _needs_g1 and supports is None:
            if n == 4:
                from geometrykit.runtimes._npu import patch_np

                surface, _corners = patch_np.discrete_coons_patch(
                    curves, tol=1e-6, degree=3
                )
                return surface
            raise NotImplementedError(
                "pure patch supports exactly four boundary curves for now "
                "(n-sided chart machinery is a later increment)"
            )

        # Auto‑align to a closed loop: curve[i].end ≈ curve[i+1].start
        curves = list(curves)
        eps = 1e-4
        for i in range(n):
            j = (i + 1) % n
            d_end_start = (curves[i].point_at(1) - curves[j].point_at(0)).length()
            d_start_start = (curves[i].point_at(0) - curves[j].point_at(0)).length()
            if d_end_start > eps and d_start_start < eps:
                curves[i] = curves[i].reverse()

        require_occ()

        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
        from OCC.Core.Geom import Geom_BSplineSurface
        from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_G1
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopExp import TopExp_Explorer

        _CONST_MAP = {"C0": GeomAbs_C0, "G1": GeomAbs_G1}
        full_constraints = [
            _CONST_MAP.get(c.upper(), GeomAbs_C0) if c else GeomAbs_C0
            for c in (constraints or ["C0"] * n)
        ]
        while len(full_constraints) < n:
            full_constraints.append(GeomAbs_C0)

        if supports is None:
            supports = [None] * n
        for i, (c, s) in enumerate(zip(full_constraints, supports)):
            if c == GeomAbs_G1 and s is None:
                raise ValueError(
                    f"G1 constraint on curve[{i}] requires a support Surface "
                    "(the adjacent face that defines the tangent direction)."
                )

        patch_mid = Vec(
            sum(c.point_at(0.5).x for c in curves) / n,
            sum(c.point_at(0.5).y for c in curves) / n,
            sum(c.point_at(0.5).z for c in curves) / n,
        )

        filler = BRepOffsetAPI_MakeFilling()
        filler.SetConstrParam(tol, tol, 2 * tol, 2 * tol)

        for i, curve in enumerate(curves):
            gc = full_constraints[i]
            supp = supports[i] if supports and i < len(supports) else None
            if supp is not None:
                # the constraint edge must be the support face's own edge
                face, edge = _support_face_and_edge(supp, curve, patch_mid)
                filler.Add(edge, face, gc)
            else:
                edge = _build_occ_edge(curve)
                filler.Add(edge, gc)

        filler.Build()
        if not filler.IsDone():
            raise RuntimeError("Patch MakeFilling returned IsDone=False")

        exp = TopExp_Explorer(filler.Shape(), TopAbs_FACE)
        if not exp.More():
            raise RuntimeError("Patch produced no face")
        result_face = exp.Current()
        adaptor = BRepAdaptor_Surface(result_face)
        geom = adaptor.Surface().Surface()
        bspline = Geom_BSplineSurface.DownCast(geom)
        if bspline is None:
            raise RuntimeError("Patch result is not a BSpline surface")

        result = cls.from_occ_surface(bspline)
        result._occ_face = result_face
        return result

    @classmethod
    def sweep(
        cls,
        rails: "Sequence[Curve]",
        profiles: "Sequence[Curve]",
    ) -> "Surface":
        """Create a swept surface (1‑2 rails × 1+ profiles).

        Uses OCC ``BRepOffsetAPI_MakePipeShell``.  Not yet implemented
        — skeleton only.

        Args:
            rails:    Guide curves (1 or 2).
            profiles: Cross‑section curves (1 or more).

        Raises:
            NotImplementedError: Always (skeleton).
        """
        raise NotImplementedError("Surface.sweep() is not yet implemented")

    @classmethod
    def ribbon(
        cls,
        curve: "Curve",
        angles: "Sequence[float] | None" = None,
        width: float = 100.0,
        angle: float | None = None,
        angle_end: float | None = None,
        angle_deg: float | None = None,
        angle_end_deg: float | None = None,
        angles_deg: "Sequence[float] | None" = None,
        n_pts: int = 20,
        symmetric: bool = False,
    ) -> "Surface":
        """Create a ribbon (ruled support surface) along a curve.

        The result is used as a G1 support surface in :meth:`patch`. By
        default the ribbon shares one exact edge with *curve* and extends
        outward by *width* at the given *angles* (a one-sided ribbon, the
        spine on a boundary edge). With ``symmetric=True`` the spine runs
        down the **middle** of the ribbon, which is offset ``width / 2`` to
        either side -- the CATIA-style tangency support, where the constraint
        is read on the spine rather than on one side of it, so the same
        support serves a patch on either side of the curve.

        ``angles`` is a list of rotation angles (**radians**) around the
        curve tangent (0 = horizontal perpendicular).  With one angle
        there is no twist; with two the ribbon twists linearly from start
        to end; with more the angles are distributed at equal chord‑length
        intervals.

        For convenience use ``angle_deg`` / ``angle_end_deg`` for degrees,
        or ``angles_deg`` for a list of degrees.

        Args:
            curve:         Spine curve.
            angles:        Rotation angles around tangent (**radians**).
            width:         Ribbon width / offset distance.
            angle:         Start angle (rad).  Overridden by *angles*.
            angle_end:     End angle (rad).  Overridden by *angles*.
            angle_deg:     Start angle (degrees).
            angle_end_deg: End angle (degrees).
            angles_deg:    Angles list in degrees (overrides *angles*).
            n_pts:         Retained for API compatibility (ignored -- the pure
                            rule samples nothing).
            symmetric:     Centre the ribbon on *curve* (spine in the middle).

        Returns:
            A new Surface suitable as G1 support for :meth:`patch`.
        """
        # Resolve angles list from the various input styles
        if angles_deg is not None:
            ang_list = [math.radians(a) for a in angles_deg]
        elif angles is not None:
            ang_list = list(angles)
        elif angle_deg is not None or angle_end_deg is not None:
            a0 = math.radians(angle_deg) if angle_deg is not None else 0.0
            a1 = math.radians(angle_end_deg) if angle_end_deg is not None else a0
            ang_list = [a0, a1]
        elif angle is not None or angle_end is not None:
            a0 = angle if angle is not None else 0.0
            a1 = angle_end if angle_end is not None else a0
            ang_list = [a0, a1]
        else:
            ang_list = [0.0]

        up = Vec(0, 0, 1)

        # ── Sample curve and compute offset points ─────────────────
        def _interp_angle(t):
            n = len(ang_list)
            if n == 1:
                return ang_list[0]
            pos = t * (n - 1)
            i = int(pos)
            if i >= n - 1:
                return ang_list[-1]
            frac = pos - i
            return ang_list[i] + frac * (ang_list[i + 1] - ang_list[i])

        def _cross(t) -> tuple:
            """Point on the spine and the unit cross direction there."""
            p = curve.point_at(t)
            tan = curve.tangent_at(t)
            perp = (tan**up).normalized()
            a = _interp_angle(t)
            if abs(a) > 1e-12:
                c, s = math.cos(a), math.sin(a)
                perp = perp * c + up * s
            return p, perp

        # ── Rails (pure kernel) ─────────────────────────────────────────
        # One-sided: the spine itself is a rail, exactly as given (no
        # re-parameterisation); the far rail offsets the spine's control
        # net, so both rails share knots, degree and parameterisation.
        # Symmetric: both rails offset the control net, so the spine lies
        # exactly on the middle line v = 0.5 -- a ruled surface between
        # same-parameterised rails interpolates their average there.
        def _perp_at(t):
            _p, perp = _cross(t)
            return perp

        if symmetric:
            rail_a = _offset_control_net(curve, -width / 2.0, _perp_at)
            rail_b = _offset_control_net(curve, +width / 2.0, _perp_at)
        else:
            rail_a = curve
            rail_b = _offset_control_net(curve, width, _perp_at)

        # ── Exact ruled surface between the rails ─────────────────────────
        # Degree 1 in v over the two rails: S(u, v) = (1-v) A(u) + v B(u).
        # No sampling, no fitting, no OCC -- n_pts is retained for API
        # compatibility only. (_support_face_and_edge rebuilds an OCC face
        # on demand for the reference twin path, so no cached face is
        # needed here.)
        if (
            len(rail_a.points) != len(rail_b.points)
            or rail_a.degree != rail_b.degree
            or list(rail_a.knots) != list(rail_b.knots)
        ):
            raise ValueError("ribbon rails lost their common parameterisation")
        net_rows = [
            [Vec(pa.x, pa.y, pa.z), Vec(pb.x, pb.y, pb.z)]
            for pa, pb in zip(rail_a.points, rail_b.points)
        ]
        return cls(
            control_points=net_rows,
            uknots=list(rail_a.knots),
            vknots=[0.0, 1.0],
            umults=[int(m) for m in rail_a.multiplicities],
            vmults=[2, 2],
            udegree=rail_a.degree,
            vdegree=1,
        )


def _offset_control_net(curve: "Curve", distance: float, perp_at) -> "Curve":
    """Offset *curve* by *distance* along its cross direction, keeping its
    knot vector, degree and parameterisation.

    ``perp_at(t)`` returns the unit cross direction at parameter ``t``. Each
    control point moves by ``distance * perp_at(greville_i)``, so the offset
    rail is parameterised exactly like the spine and their average is the
    spine again.
    """
    from geometrykit.geometry.curve import Curve as _Curve

    flat: list = []
    for knot, mult in zip(curve.knots, curve.multiplicities):
        flat.extend([float(knot)] * int(mult))
    degree = curve.degree
    points = []
    for index, point in enumerate(curve.points):
        # Greville abscissa of this control point
        window = flat[index + 1 : index + degree + 1]
        t = sum(window) / len(window) if window else 0.0
        perp = perp_at(t)
        points.append(point + perp * distance)
    return _Curve(
        control_points=points,
        knots=list(curve.knots),
        multiplicities=list(curve.multiplicities),
        degree=degree,
    )


def _build_occ_edge(curve: "Curve") -> object:
    """Build an OCC ``TopoDS_Edge`` from an ifckit *Curve*."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.Geom import Geom_BSplineCurve
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TColgp import TColgp_Array1OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal

    n = len(curve.points)
    poles = TColgp_Array1OfPnt(1, n)
    for i, p in enumerate(curve.points):
        poles.SetValue(i + 1, gp_Pnt(p.x, p.y, p.z))

    ukn = TColStd_Array1OfReal(1, len(curve.knots))
    for i, k in enumerate(curve.knots):
        ukn.SetValue(i + 1, k)
    mul = TColStd_Array1OfInteger(1, len(curve.multiplicities))
    for i, m in enumerate(curve.multiplicities):
        mul.SetValue(i + 1, m)

    if curve.rational:
        wgt = TColStd_Array1OfReal(1, n)
        for i, w in enumerate(curve._weights):
            wgt.SetValue(i + 1, w)
        geom = Geom_BSplineCurve(poles, wgt, ukn, mul, curve.degree, curve.closed)
    else:
        geom = Geom_BSplineCurve(poles, ukn, mul, curve.degree, curve.closed)

    return BRepBuilderAPI_MakeEdge(geom, curve.knots[0], curve.knots[-1]).Edge()


def _build_occ_curve(curve: "Curve") -> object:
    """Build an OCC ``Geom_BSplineCurve`` from an ifckit *Curve* (no edge wrapper)."""
    from OCC.Core.Geom import Geom_BSplineCurve
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TColgp import TColgp_Array1OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal

    n = len(curve.points)
    poles = TColgp_Array1OfPnt(1, n)
    for i, p in enumerate(curve.points):
        poles.SetValue(i + 1, gp_Pnt(p.x, p.y, p.z))

    ukn = TColStd_Array1OfReal(1, len(curve.knots))
    for i, k in enumerate(curve.knots):
        ukn.SetValue(i + 1, k)
    mul = TColStd_Array1OfInteger(1, len(curve.multiplicities))
    for i, m in enumerate(curve.multiplicities):
        mul.SetValue(i + 1, m)

    if curve.rational:
        wgt = TColStd_Array1OfReal(1, n)
        for i, w in enumerate(curve._weights):
            wgt.SetValue(i + 1, w)
        return Geom_BSplineCurve(poles, wgt, ukn, mul, curve.degree, curve.closed)

    return Geom_BSplineCurve(poles, ukn, mul, curve.degree, curve.closed)


def _curve_from_occ_bspline(bspline: object) -> "Curve":
    """Extract an ifckit ``Curve`` from an OCC ``Geom_BSplineCurve``."""
    from geometrykit.geometry.curve import Curve as _Curve

    poles = [
        Vec(bspline.Pole(i + 1).X(), bspline.Pole(i + 1).Y(), bspline.Pole(i + 1).Z())
        for i in range(bspline.NbPoles())
    ]
    knots = [bspline.Knot(i + 1) for i in range(bspline.NbKnots())]
    mults = [bspline.Multiplicity(i + 1) for i in range(bspline.NbKnots())]
    degree = bspline.Degree()
    weights = (
        [bspline.Weight(i + 1) for i in range(bspline.NbPoles())] if bspline.IsRational() else None
    )
    closed = bspline.IsPeriodic()

    return _Curve(
        control_points=poles,
        knots=knots,
        multiplicities=mults,
        degree=degree,
        weights=weights,
        closed=closed,
    )


# ---------------------------------------------------------------------------
# OCC evaluation helpers
# ---------------------------------------------------------------------------


def _build_occ_surface(surf: Surface):
    """Rebuild an OCC ``Geom_BSplineSurface`` from an ifckit Surface."""
    from OCC.Core.gp import gp_Pnt

    nu, nv = surf.nu, surf.nv
    poles = TColgp_Array2OfPnt(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            p = surf.control_points[i][j]
            poles.SetValue(i + 1, j + 1, gp_Pnt(p.x, p.y, p.z))

    def _uniq_knots(knots: list[float]) -> list[float]:
        """Extract unique knots from a full knot vector (with multiplicities)."""
        uniq: list[float] = []
        for k in knots:
            if not uniq or abs(k - uniq[-1]) > 1e-10:
                uniq.append(k)
        return uniq

    ukn = TColStd_Array1OfReal(1, len(surf.uknots))
    for i, k in enumerate(surf.uknots):
        ukn.SetValue(i + 1, k)
    vkn = TColStd_Array1OfReal(1, len(surf.vknots))
    for i, k in enumerate(surf.vknots):
        vkn.SetValue(i + 1, k)

    # OCC Geom_BSplineSurface expects unique knots + multiplicities separately.
    # ifckit stores the full knot vector; convert here.
    ukn_uniq = _uniq_knots(surf.uknots)
    vkn_uniq = _uniq_knots(surf.vknots)
    ukn_arr = TColStd_Array1OfReal(1, len(ukn_uniq))
    for i, k in enumerate(ukn_uniq):
        ukn_arr.SetValue(i + 1, k)
    vkn_arr = TColStd_Array1OfReal(1, len(vkn_uniq))
    for i, k in enumerate(vkn_uniq):
        vkn_arr.SetValue(i + 1, k)

    um = TColStd_Array1OfInteger(1, len(surf.umults))
    for i, m in enumerate(surf.umults):
        um.SetValue(i + 1, m)
    vm = TColStd_Array1OfInteger(1, len(surf.vmults))
    for i, m in enumerate(surf.vmults):
        vm.SetValue(i + 1, m)

    if surf.rational:
        wgt = TColStd_Array2OfReal(1, nu, 1, nv)
        for i in range(nu):
            for j in range(nv):
                wgt.SetValue(i + 1, j + 1, surf._weights[i][j])
        args = (
            poles,
            wgt,
            ukn_arr,
            vkn_arr,
            um,
            vm,
            surf.udegree,
            surf.vdegree,
            surf.uclosed,
            surf.vclosed,
        )
        return Geom_BSplineSurface(*args)

    args = (poles, ukn_arr, vkn_arr, um, vm, surf.udegree, surf.vdegree, surf.uclosed, surf.vclosed)
    return Geom_BSplineSurface(*args)


def occ_eval_point(surf: Surface, u: float, v: float) -> Vec:
    """Evaluate surface point at ``(u, v)`` — both in ``[0, 1]``."""
    require_occ()
    pt = _build_occ_surface(surf).Value(u, v)
    return Vec(pt.X(), pt.Y(), pt.Z())


def occ_eval_tangents(surf: Surface, u: float, v: float) -> "tuple[Vec, Vec]":
    """Evaluate tangents at specified parameters using OCC."""
    require_occ()
    occ = _build_occ_surface(surf)
    d1u, d1v = gp_Vec(), gp_Vec()
    occ.D1(u, v, gp_Pnt(), d1u, d1v)
    return Vec(d1u.X(), d1u.Y(), d1u.Z()), Vec(d1v.X(), d1v.Y(), d1v.Z())


def occ_normal_at(surf: Surface, u: float, v: float) -> Vec:
    """Exact surface normal at ``(u, v)`` from OCC first derivatives.

    Orientation follows the parametrisation (not forced z-up) — callers
    orient explicitly where a convention applies. Requires ``pythonocc-core``.
    """
    tu, tv = occ_eval_tangents(surf, u, v)
    return (tu**tv).normalized()


def occ_project_point(surf: Surface, pt: Vec) -> "tuple[Vec, tuple[float, float]]":
    """Closest point on the surface + its ``(u, v)`` parameters.

    Uses OCC ``GeomAPI_ProjectPointOnSurf`` on the mathematical surface
    (exact up to OCC precision, independent of any tessellation).
    Requires ``pythonocc-core``.

    Args:
        surf: Surface to project onto.
        pt:   World point to project.

    Returns:
        ``(projected Vec, (u, v))``. Raises ``ValueError`` when no
        projection exists.
    """
    require_occ()
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf

    proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(pt.x, pt.y, pt.z), _build_occ_surface(surf))
    if proj.NbPoints() < 1:
        raise ValueError("No projection of point onto surface found")
    q = proj.NearestPoint()
    u, v = proj.LowerDistanceParameters()
    return Vec(q.X(), q.Y(), q.Z()), (u, v)


def occ_face_bounds(surf: Surface) -> "tuple[float, float, float, float]":
    """``(Umin, Umax, Vmin, Vmax)`` of the trimmed face.

    Uses the cached OCC face (``Surface.fill``/``patch`` keep it); falls back
    to the native knot range when there is no face or no ``pythonocc-core``.
    """
    face = getattr(surf, "_occ_face", None)
    if face is not None and _HAS_OCC:
        import warnings

        from OCC.Core.BRepTools import breptools_UVBounds

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # deprecated in pythonocc-core ≥ 7.7
            try:
                return tuple(breptools_UVBounds(face))
            except Exception:
                pass
    return (surf.uknots[0], surf.uknots[-1], surf.vknots[0], surf.vknots[-1])


def occ_trim_loops(surf: Surface, samples: int = 64) -> "List[List[tuple[float, float]]]":
    """Trim loops of the face as UV polylines.

    Reads each edge p-curve (``BRep_Tool.CurveOnSurface``) from the cached OCC
    face; edges without a p-curve are skipped. Without a cached face (or
    without ``pythonocc-core``) falls back to the four native boundary
    edges as separate loops (south, east, north, west).
    Loop orientation is OCC edge order — callers must not rely on it
    (see ``field._classify`` for position-based identification).

    Args:
        surf:    Surface whose face trim to read.
        samples: Points per edge (excluding closure duplication).

    Returns:
        List of loops; each loop a list of ``(u, v)`` tuples.
    """
    u0, u1, v0, v1 = surf.uknots[0], surf.uknots[-1], surf.vknots[0], surf.vknots[-1]
    fallback = [
        [(u0, v0), (u1, v0)],
        [(u1, v0), (u1, v1)],
        [(u1, v1), (u0, v1)],
        [(u0, v1), (u0, v0)],
    ]
    face = getattr(surf, "_occ_face", None)
    if face is None or not _HAS_OCC:
        return fallback
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer

    loops: "List[List[tuple[float, float]]]" = []
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        try:
            h, first, last = BRep_Tool.CurveOnSurface(exp.Current(), face)
        except Exception:
            exp.Next()
            continue
        loop = []
        for k in range(samples + 1):
            p = h.Value(first + (last - first) * k / samples)
            loop.append((p.X(), p.Y()))
        loops.append(loop)
        exp.Next()
    return loops or fallback


def occ_intersect_plane(surf: Surface, plane: "Plane") -> List["Curve"]:
    """Intersect a surface with a plane → list of ``Curve``.

    Uses OCC ``BRepAlgoAPI_Section``.  Requires ``pythonocc-core``.
    """
    require_occ()
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.GeomAbs import GeomAbs_BSplineCurve

    from geometrykit.geometry.curve import Curve as _Curve

    occ_surf = _build_occ_surface(surf)
    face = BRepBuilderAPI_MakeFace(occ_surf, 1e-6).Face()

    pl = plane
    occ_plane = gp_Pln(
        gp_Pnt(pl.origin.x, pl.origin.y, pl.origin.z), gp_Dir(pl.z_axis.x, pl.z_axis.y, pl.z_axis.z)
    )

    section = BRepAlgoAPI_Section(face, occ_plane)
    section.Build()
    shape = section.Shape()

    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer

    result: List = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = exp.Current()
        adaptor = BRepAdaptor_Curve(edge)
        if adaptor.GetType() == GeomAbs_BSplineCurve:
            result.append(_Curve.from_occ_edge(edge))
        exp.Next()
    return result


# ---------------------------------------------------------------------------
# Tessellation
# ---------------------------------------------------------------------------


def _face_edge_on(face, point: Vec):
    """The edge of *face* whose midpoint is closest to *point*."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer

    best = None
    best_distance = None
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while explorer.More():
        edge = explorer.Current()
        handle = BRep_Tool.Curve(edge)[0]
        if handle is not None:
            first, last = BRep_Tool.Curve(edge)[1], BRep_Tool.Curve(edge)[2]
            middle = handle.Value((first + last) / 2.0)
            gap = (
                (middle.X() - point.x) ** 2
                + (middle.Y() - point.y) ** 2
                + (middle.Z() - point.z) ** 2
            )
            if best_distance is None or gap < best_distance:
                best, best_distance = edge, gap
        explorer.Next()
    if best is None:
        raise RuntimeError("support face has no edge to attach the constraint to")
    return best


def _support_face_and_edge(supp: Surface, curve: "Curve", towards: Vec):
    """The support face that carries *curve* on its boundary, and that edge.

    One-sided ribbons arrive with a cached (face, edge) pair whose edge is the
    spine. A symmetric ribbon carries the spine down the middle, so the face is
    bounded at the spine -- the half on the side of *towards*, which has the
    same tangent plane along the spine -- and the resulting edge is returned.
    MakeFilling needs the face's own edge, not a rebuilt one.
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

    cached_face = getattr(supp, "_occ_face", None)
    cached_edge = getattr(supp, "_occ_edge", None)
    if cached_face is not None and cached_edge is not None:
        return cached_face, cached_edge

    geometry = _build_occ_surface(supp)
    full = BRepBuilderAPI_MakeFace(geometry, 1e-6).Face()
    bounds = BRepAdaptor_Surface(full)
    u0, u1 = bounds.FirstUParameter(), bounds.LastUParameter()
    v0, v1 = bounds.FirstVParameter(), bounds.LastVParameter()

    middle = curve.point_at(0.5)
    _projected, (_u, v_spine) = supp.project(middle)
    span = max(abs(v1 - v0), 1e-12)
    if min(abs(v_spine - v0), abs(v_spine - v1)) < 1e-6 * span:
        face = full
    else:
        u_mid = (u0 + u1) / 2.0

        def _mid_point(v_lo, v_hi) -> Vec:
            point = bounds.Value(u_mid, (v_lo + v_hi) / 2.0)
            return Vec(point.X(), point.Y(), point.Z())

        lower = _mid_point(v0, v_spine)
        upper = _mid_point(v_spine, v1)
        if abs(lower - towards) <= abs(upper - towards):
            face = BRepBuilderAPI_MakeFace(geometry, u0, u1, v0, v_spine, 1e-6).Face()
        else:
            face = BRepBuilderAPI_MakeFace(geometry, u0, u1, v_spine, v1, 1e-6).Face()
    return face, _face_edge_on(face, middle)


def _build_occ_face(surf: Surface) -> object:
    """Build an OCC ``TopoDS_Face`` from an ifckit Surface."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace

    return BRepBuilderAPI_MakeFace(_build_occ_surface(surf), 1e-6).Face()


def occ_tessellate(
    surf: Surface, deflection: float | TessellationDetail = 0.01
) -> "tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]":
    """Triangulate a surface → ``(vertices, triangles)``.

    *vertices* is a list of ``(x, y, z)`` tuples.
    *triangles* is a list of 1‑based index triples for
    ``IfcTriangulatedFaceSet.CoordIndex``.

    Requires ``pythonocc-core``.
    """
    require_occ()
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location

    # Accept TessellationDetail as deflection shorthand
    try:
        from ifckit.schema import TessellationDetail as _TD

        _TESS_DETAIL_MAP = {
            _TD.COARSE: 0.05,
            _TD.MEDIUM: 0.02,
            _TD.FINE: 0.01,
            _TD.ULTRA: 0.005,
        }
        if isinstance(deflection, _TD):
            deflection = _TESS_DETAIL_MAP[deflection]
    except ImportError:
        pass

    deflection = float(deflection)

    face = getattr(surf, "_occ_face", None)
    if face is None:
        face = _build_occ_face(surf)
    mesh = BRepMesh_IncrementalMesh(face, deflection)
    mesh.Perform()

    vertices: "list[tuple[float, float, float]]" = []
    triangles: "list[tuple[int, int, int]]" = []
    offset = 0

    exp = TopExp_Explorer(face, TopAbs_FACE)
    while exp.More():
        f = exp.Current()
        loc = TopLoc_Location()
        T = BRep_Tool.Triangulation(f, loc)
        if T is not None:
            nb_nodes = T.NbNodes()
            for i in range(1, nb_nodes + 1):
                p = T.Node(i)
                vertices.append((p.X(), p.Y(), p.Z()))

            nb_tri = T.NbTriangles()
            for i in range(1, nb_tri + 1):
                t = T.Triangle(i)
                i1, i2, i3 = t.Get()
                triangles.append((offset + i1, offset + i2, offset + i3))

            offset += nb_nodes

        exp.Next()

    return vertices, triangles


def occ_tessellate_to_ifc(
    surf: Surface,
    ifc_file: "ifcopenshell.file",
    deflection: "float | object" = 0.01,
) -> "ifcopenshell.entity_instance":
    """Triangulate a surface and wrap in ``IfcTriangulatedFaceSet``.

    Accepts the same *deflection* options as :func:`occ_tessellate`
    (float or ``TessellationDetail``).

    Requires ``pythonocc-core``.
    """
    verts, tris = occ_tessellate(surf, deflection)
    coord_list = ifc_file.create_entity("IfcCartesianPointList3D", CoordList=verts)
    return ifc_file.create_entity(
        "IfcTriangulatedFaceSet",
        Coordinates=coord_list,
        CoordIndex=tris,
    )
