# This file was generated with the assistance of an AI coding tool.
"""Interchange writers for kernel geometry (occ-free except where noted).

- ``surface_to_obj`` / ``write_obj`` -- occ-free quad mesh; every 3D viewer
  and Blender read it.
- ``write_stl`` -- the same mesh as ASCII STL triangles.
- ``write_step_surfaces`` -- AP214 STEP with real
  ``B_SPLINE_SURFACE_WITH_KNOTS`` faces, which is what CAD kernels
  (Rhino, CATIA, SolidWorks) import. No host dependency, no tessellation:
  the control net round-trips exactly.
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


def write_obj(
    surfaces: list, path: str, labels: list | None = None, nu: int = 30, nv: int = 30
) -> str:
    """Write one or more surfaces into a single OBJ file (quads).

    ``nu``/``nv`` are the sampling resolutions per surface -- raise them
    for a finer mesh in a viewer or Blender.
    """
    labels = labels or [f"patch_{i}" for i in range(len(surfaces))]
    lines = ["# geometrykit flowing-roof output (occ-free kernel)"]
    vertex_offset = 0
    for surf, label in zip(surfaces, labels):
        verts, faces = surface_to_obj(surf, nu=nu, nv=nv, label=label)
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


def write_stl(surfaces: list, path: str, nu: int = 30, nv: int = 30) -> str:
    """Write surfaces as ASCII STL (triangles from the quad grid)."""
    verts = []
    faces = []
    offset = 0
    for surf in surfaces:
        v, f = surface_to_obj(surf, nu=nu, nv=nv)
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


def _step_knot_list(knots, mults):
    """Part-21 knot list text: one entry per distinct knot.

    ``b_spline_*_with_knots`` pairs a knot list with a multiplicity list of
    the *same length* -- the distinct knots, not the expanded knot vector.
    (Emitting the expanded vector makes the reader reject the surface.)
    """
    values = list(knots)
    if len(values) != len(list(mults)):
        raise ValueError(
            f"knot list ({len(values)}) and multiplicity list "
            f"({len(list(mults))}) must have one entry per distinct knot"
        )
    return ",".join(repr(float(value)) for value in values)


def _step_mult_list(mults) -> str:
    """Part-21 integer list text (multiplicities)."""
    return ",".join(str(int(m)) for m in mults)


def _wrap_entity(line: str, width: int = 72) -> str:
    """Wrap one part-21 statement at comma boundaries.

    Long statements are legal, but readers (OCCT's included) are happier with
    short physical lines, so every continuation is indented by one space --
    the conventional layout for an exchange file.
    """
    if len(line) <= width:
        return line
    pieces = []
    current = ""
    depth = 0
    for char in line:
        current += char
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if len(current) >= width and char == "," and depth > 0:
            pieces.append(current)
            current = ""
    if current:
        pieces.append(current)
    return (chr(10) + " ").join(pieces)


class _StepFile:
    """Minimal AP214 writer: entities in reference order, ids from 1."""

    def __init__(self) -> None:
        self._lines: list = []
        self._next = 1

    def add(self, body: str) -> int:
        ident = self._next
        self._next += 1
        self._lines.append(f"#{ident}={body};")
        return ident

    def text(self, name: str) -> str:
        header = chr(10).join(
            [
                "ISO-10303-21;",
                "HEADER;",
                "FILE_DESCRIPTION((''),'2;1');",
                f"FILE_NAME('{name}','1970-01-01T00:00:00',(''),(''),"
                "'geometrykit','geometrykit','');",
                "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));",
                "ENDSEC;",
                "DATA;",
            ]
        )
        body = chr(10).join(_wrap_entity(line) for line in self._lines)
        return (header + chr(10) + body + chr(10) + "ENDSEC;"
                + chr(10) + "END-ISO-10303-21;" + chr(10))

    def write(self, path) -> None:
        from pathlib import Path

        Path(path).write_text(self.text("geometrykit patch export"))


def _step_point(step: _StepFile, point) -> int:
    x, y, z = (repr(float(c)) for c in point)
    return step.add(f"CARTESIAN_POINT('',({x},{y},{z}))")


def _step_curve(step: _StepFile, cps, degree, knots, mults, label) -> int:
    ids = [_step_point(step, p) for p in cps]
    rows = ",".join("#" + str(i) for i in ids)
    return step.add(
        f"B_SPLINE_CURVE_WITH_KNOTS('{label}',{int(degree)},({rows}),.UNSPECIFIED.,"
        f".F.,.F.,({_step_mult_list(mults)}),({_step_knot_list(knots, mults)}),.UNSPECIFIED.)"
    )


def _step_point2d(step: _StepFile, x: float, y: float) -> int:
    return step.add(f"CARTESIAN_POINT('',({repr(float(x))},{repr(float(y))}))")


def _step_axis(step: _StepFile, label: str = "") -> int:
    """A minimal AXIS2_PLACEMENT_3D at the origin (representation anchor)."""
    origin = _step_point(step, (0.0, 0.0, 0.0))
    z_dir = step.add("DIRECTION('',(0.,0.,1.))")
    x_dir = step.add("DIRECTION('',(1.,0.,0.))")
    return step.add(f"AXIS2_PLACEMENT_3D('{label}',#{origin},#{z_dir},#{x_dir})")


def _step_curve(step: _StepFile, cps, degree, knots, mults, label) -> int:
    ids = [_step_point(step, p) for p in cps]
    rows = ",".join(f"#{i}" for i in ids)
    return step.add(
        f"B_SPLINE_CURVE_WITH_KNOTS('{label}',{int(degree)},({rows}),.UNSPECIFIED.,"
        f".F.,.F.,({_step_mult_list(mults)}),({_step_knot_list(knots, mults)}),.UNSPECIFIED.)"
    )


def _step_edge(step, label, curve, param_origin, param_dir, context, surface_id) -> int:
    """An EDGE_CURVE whose geometry carries its own pcurve.

    The 3D boundary curve of a tensor-product surface *is* the surface
    restricted to that edge, and it inherits the surface's parametrization,
    so the pcurve is exactly the parameter-space segment the edge runs along
    -- no projection, no approximation. The parameter context and the
    definitional representation follow the shape OCCT itself writes.
    """
    origin = _step_point2d(step, param_origin[0], param_origin[1])
    direction = step.add(f"DIRECTION('',({repr(float(param_dir[0]))},{repr(float(param_dir[1]))}))")
    vector = step.add(f"VECTOR('',#{direction},1.)")
    line = step.add(f"LINE('',#{origin},#{vector})")
    representation = step.add(
        f"DEFINITIONAL_REPRESENTATION('',(#{line}),#{context})"
    )
    # the associated geometry of a surface_curve must be a PCURVE (which
    # names its basis surface), not a bare definitional representation
    pcurve = step.add(f"PCURVE('',#{surface_id},#{representation})")
    return step.add(f"SURFACE_CURVE('{label}',#{curve},(#{pcurve}),.PCURVE_S1.)")


def _step_face(step: _StepFile, surface, label: str, context, param_context) -> int:
    """One ADVANCED_FACE over *surface*, bounded by its four boundary curves.

    The boundary of a tensor-product B-spline surface is itself a B-spline
    curve -- the boundary control row with the surface's knot vector in the
    varying direction -- so the edges are exact, not approximated, and the
    four corner poles are shared as STEP vertices.
    """
    cps = [[(p.x, p.y, p.z) for p in row] for row in surface.control_points]
    point_ids = [[_step_point(step, p) for p in row] for row in cps]
    # control_points_list is a LIST OF ROWS: ((#p,#p),(#p,#p),...)
    rows = "(" + ",".join(
        "(" + ",".join(f"#{i}" for i in row) + ")" for row in point_ids
    ) + ")"
    surf = step.add(
        f"B_SPLINE_SURFACE_WITH_KNOTS('{label}',{int(surface.udegree)},"
        f"{int(surface.vdegree)},{rows},.UNSPECIFIED.,.F.,.F.,.F.,"
        f"({_step_mult_list(surface.umults)}),({_step_mult_list(surface.vmults)}),"
        f"({_step_knot_list(surface.uknots, surface.umults)}),"
        f"({_step_knot_list(surface.vknots, surface.vmults)}),.UNSPECIFIED.)"
    )

    n_u = len(cps)
    n_v = len(cps[0])
    corner_points = {
        (0, 0): point_ids[0][0],
        (1, 0): point_ids[n_u - 1][0],
        (1, 1): point_ids[n_u - 1][n_v - 1],
        (0, 1): point_ids[0][n_v - 1],
    }
    vertices = {
        key: step.add(f"VERTEX_POINT('',#{point})")
        for key, point in corner_points.items()
    }

    # Every boundary curve is emitted in its *natural* order -- reversing a
    # control row without reversing the knot vector would produce a curve
    # that is not the surface's boundary at all. The loop's direction is
    # carried by the ORIENTED_EDGE sense flags instead, and each pcurve is
    # the parameter-space segment matching its curve's own parametrization.
    edges = (
        # name, poles in natural order, degree, knots, mults,
        # (edge_start corner, edge_end corner), pcurve origin, pcurve dir, sense
        ("south", [cps[i][0] for i in range(n_u)], surface.udegree,
         surface.uknots, surface.umults, (0, 0), (1, 0),
         (0.0, 0.0), (1.0, 0.0), True),
        ("east", [cps[n_u - 1][j] for j in range(n_v)], surface.vdegree,
         surface.vknots, surface.vmults, (1, 0), (1, 1),
         (1.0, 0.0), (0.0, 1.0), True),
        ("north", [cps[i][n_v - 1] for i in range(n_u)], surface.udegree,
         surface.uknots, surface.umults, (0, 1), (1, 1),
         (0.0, 1.0), (1.0, 0.0), False),
        ("west", [cps[0][j] for j in range(n_v)], surface.vdegree,
         surface.vknots, surface.vmults, (0, 0), (0, 1),
         (0.0, 0.0), (0.0, 1.0), False),
    )
    oriented = []
    for (name, row, degree, knots, mults, start_key, end_key,
         origin, direction, sense) in edges:
        curve = _step_curve(step, row, degree, knots, mults, f"{label}_{name}")
        geometry = _step_edge(
            step, f"{label}_{name}", curve, origin, direction, param_context, surf
        )
        edge = step.add(
            f"EDGE_CURVE('',#{vertices[start_key]},#{vertices[end_key]},#{geometry},.T.)"
        )
        flag = ".T." if sense else ".F."
        oriented.append(step.add(f"ORIENTED_EDGE('',*,*,#{edge},{flag})"))

    loop = step.add(f"EDGE_LOOP('',({','.join('#' + str(o) for o in oriented)}))")
    bound = step.add(f"FACE_OUTER_BOUND('',#{loop},.T.)")
    return step.add(f"ADVANCED_FACE('{label}',(#{bound}),#{surf},.T.)")


def write_step_surfaces(surfaces: list, path, labels=None) -> str:
    """Write NURBS surfaces to an AP214 STEP file (manifold surface B-rep).

    Real ``B_SPLINE_SURFACE_WITH_KNOTS`` inside ``ADVANCED_FACE``s, wrapped in
    the ``SHELL_BASED_SURFACE_MODEL`` / ``MANIFOLD_SURFACE_SHAPE_REPRESENTATION``
    chain that CAD kernels (Rhino, CATIA, SolidWorks, OCCT) read. Each face is
    bounded by the surface's own four boundary curves with their pcurves, so
    nothing is approximated: the control net round-trips exactly and a reader
    reconstructs the same knots, poles and geometry.

    Args:
        surfaces: list of `Surface` (rational weights are rejected rather
            than silently dropped -- the exchange needs weighted poles).
        path:     output file.
        labels:   optional names, one per surface.

    Returns:
        The STEP text that was written.
    """
    labels = labels or [f"Patch_{i}" for i in range(len(surfaces))]
    for surface, label in zip(surfaces, labels):
        if getattr(surface, "rational", False):
            raise ValueError(
                f"STEP export of the rational surface '{label}' is not supported "
                "(the exchange format needs weighted poles)"
            )

    step = _StepFile()
    context = step.add("APPLICATION_CONTEXT('automotive design')")
    step.add(
        "APPLICATION_PROTOCOL_DEFINITION('international standard',"
        f"'automotive_design',2000,#{context})"
    )
    product_context = step.add(f"PRODUCT_CONTEXT('',#{context},'mechanical')")
    product = step.add("PRODUCT('geometrykit','geometrykit','',(#" + str(product_context) + "))")
    formation = step.add(f"PRODUCT_DEFINITION_FORMATION('','',#{product})")
    definition_context = step.add(
        f"PRODUCT_DEFINITION_CONTEXT('part definition',#{context},'design')"
    )
    definition = step.add(f"PRODUCT_DEFINITION('design','',#{formation},#{definition_context})")
    shape = step.add(f"PRODUCT_DEFINITION_SHAPE('','',#{definition})")

    length_unit = step.add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.))")
    angle_unit = step.add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    solid_unit = step.add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    uncertainty = step.add(
        f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#{length_unit},"
        "'distance_accuracy_value','')"
    )
    geom_context = step.add(
        "(GEOMETRIC_REPRESENTATION_CONTEXT(3)"
        f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty}))"
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length_unit},#{angle_unit},#{solid_unit}))"
        "REPRESENTATION_CONTEXT('','3D Context with UNIT and UNCERTAINTY'))"
    )
    param_context = step.add(
        "(GEOMETRIC_REPRESENTATION_CONTEXT(2)"
        "PARAMETRIC_REPRESENTATION_CONTEXT()"
        "REPRESENTATION_CONTEXT('2D SPACE',''))"
    )

    axis = _step_axis(step)
    faces = [
        _step_face(step, surface, label, geom_context, param_context)
        for surface, label in zip(surfaces, labels)
    ]
    shell = step.add(f"OPEN_SHELL('',({','.join('#' + str(f) for f in faces)}))")
    model = step.add(f"SHELL_BASED_SURFACE_MODEL('',(#{shell}))")
    rep = step.add(
        f"MANIFOLD_SURFACE_SHAPE_REPRESENTATION('geometrykit_patches',"
        f"(#{axis},#{model}),#{geom_context})"
    )
    step.add(f"SHAPE_DEFINITION_REPRESENTATION(#{shape},#{rep})")

    text = step.text("geometrykit patch export")
    if "#{" in text:
        raise AssertionError("unformatted placeholder left in the STEP output")
    step.write(path)
    return text
