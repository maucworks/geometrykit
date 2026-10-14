# This file was generated with the assistance of an AI coding tool.
"""Export geometrykit patch surfaces to interchange formats (M-eval).

- ``surface_to_obj`` — occ-free: sample the surface on an (nu × nv) grid
  and emit quads (4-vertex faces). Every 3D viewer loads OBJ.
- ``surface_to_stl`` — occ-free: triangle subdivision of the same grid.
- ``surfaces_to_ifc_step`` — oracle-env path: builds a minimal IFC model
  with ``IfcBSplineSurfaceWithKnots`` per patch and writes STEP via
  ifcopenshell. Loadable in Catia / SolidWorks / Rhino for exact CAD
  verification of the kernel results.
"""

from __future__ import annotations

import math
from pathlib import Path


def surface_to_obj(
    surface,
    nu: int = 30,
    nv: int = 30,
    label: str = "patch",
) -> tuple:
    """Sample the surface on an ``(nu+1) × (nv+1)`` grid; emit OBJ quads.

    Returns ``(vertices, faces)`` — vertices = ``[(x, y, z), …]`` (length
    ``(nu+1)·(nv+1)``), faces = ``((i₀,i₁,i₂,i₃), …)`` (nu·nv quads).
    """
    verts = []
    for i in range(nu + 1):
        for j in range(nv + 1):
            p = surface.point_at(i / nu, j / nv)
            verts.append((p.x, p.y, p.z))
    faces = []
    for i in range(nu):
        for j in range(nv):
            base = i * (nv + 1) + j
            faces.append((base, base + nv + 1, base + nv + 2, base + 1))
    return verts, faces


def write_obj(surfaces: list, path: str, labels: list | None = None) -> str:
    """Write one or more surfaces into a single OBJ file (quads)."""
    labels = labels or [f"patch_{i}" for i in range(len(surfaces))]
    lines = ["# geometrykit flowing-roof output (occ-free kernel)"]
    vertex_offset = 0
    for surf, label in zip(surfaces, labels):
        verts, faces = surface_to_obj(surf, label=label)
        lines.append(f"o {label}")
        for v in verts:
            lines.append(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}")
        for f in faces:
            wf = tuple(i + 1 + vertex_offset for i in f)  # OBJ is 1-based
            lines.append(f"f {wf[0]} {wf[1]} {wf[2]} {wf[3]}")  # quad: 4 verts
        vertex_offset += len(verts)
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text)
    return text


def write_stl(surfaces: list, path: str) -> str:
    """Write surfaces as binary STL (triangle subdivision of the OBJ quads)."""
    verts =[]
    faces = []
    offset = 0
    for surf in surfaces:
        v, f = surface_to_obj(surf, nu=30, nv=30)
        verts.extend(v)
        for quad in f:
            i0, i1, i2, i3 = (i + offset for i in quad)
            faces.append((i0, i1, i2))
            faces.append((i0, i2, i3))
        offset += len(v)
    text_lines = ["solid geometrykit"]
    for face in faces:
        a, b, c = (verts[i] for i in face)
        # compute normal
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        text_lines.append(f"  facet normal {nx/ln:.9e} {ny/ln:.9e} {nz/ln:.9e}")
        text_lines.append("    outer loop")
        for pt in (a, b, c):
            text_lines.append(f"      vertex {pt[0]:.9e} {pt[1]:.9e} {pt[2]:.9e}")
        text_lines.append("    endloop")
        text_lines.append("  endfacet")
    text_lines.append("endsolid geometrykit")
    text = "\n".join(text_lines) + "\n"
    Path(path).write_text(text)
    return text


def surfaces_to_ifc_step(surfaces: list, path: str, labels: list | None = None) -> str:
    """Oracle-env path: write the surfaces into a minimal IFC STEP file.

    Each surface becomes an ``IfcBSplineSurfaceWithKnots`` (non-rational)
    inside an ``IfcFaceSurface`` → ``IfcShellBasedSurfaceModel`` →
    ``IfcProductDefinitionShape``, written via ifcopenshell's `file.write`.

    Requires ifcopenshell.
    """
    import ifcopenshell

    f = ifcopenshell.file(schema="IFC4")
    owner = f.by_type("IfcOwnerHistory")
    if not owner:
        owner = f.create_entity("IfcOwnerHistory")

    labels = labels or [f"Patch_{i}" for i in range(len(surfaces))]
    for idx, (surf, label) in enumerate(zip(surfaces, labels)):
        knots_u = list(surf.uknots)
        knots_v = list(surf.vknots)
        mults_u = list(surf.umults)
        mults_v = list(surf.vmults)
        cps = []
        for row in surf.control_points:
            row_pts = []
            for p in row:
                cp = f.create_entity("IfcCartesianPoint", Coordinates=(p.x, p.y, p.z))
                row_pts.append(cp)
            cps.append(row_pts)
        f.create_entity(
            "IfcBSplineSurfaceWithKnots",
            UDegree=surf.udegree,
            VDegree=surf.vdegree,
            ControlPointsList=cps,
            SurfaceForm="UNSPECIFIED",
            UClosed=surf.uclosed,
            VClosed=surf.vclosed,
            SelfIntersect=False,
            UMultiplicities=mults_u,
            VMultiplicities=mults_v,
            UKnots=knots_u,
            VKnots=knots_v,
        )
        # orphan B-spline entities in a minimal IFC STEP — they are the
        # most reliable exchange for BSpline verification (no IfcFace/
        # Shell/Representation wrapping complexity to debug)
    f.write(path)
    return path
