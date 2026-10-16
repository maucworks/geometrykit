# This file was generated with the assistance of an AI coding tool.
# Ported from ifckit/ifckit/geometry/__init__.py (LGPL-3.0-or-later heritage;
# ifckit is authored by the same copyright holder). The IFC
# serialisation layer (ifc_geom) stays in ifckit by design; the panel
# metrics (panel) were promoted here per the geometry-ownership plan.

"""
geometrykit.geometry
====================

Framework-agnostic geometry primitives. No Rhino, no Grasshopper, no IFC,
no external dependencies beyond the standard library.

Modules
-------
primitives  — Vec, Plane, Line, Arc, Polyline
path        — Path, assemble_path
biarc       — solve_biarc, fit_biarcs
curve       — Curve
surface     — Surface (eval/normal/tessellate kernels; OCC oracle-compare only)
field       — Field, FieldPoint, trim_grid, crossing_grid, divide_trim_edge
patterns    — quad/diamond/triangle/wave/anglebox/brick/truchet/hex/hankin cells
panel       — FieldReport, cells_report, field_report (panel metrics)
intersection — Intersection
subdivision — catmull_clark, extract_patches, write_obj
frames      — FrameField, transport_frames, fixed_ref_frames, upvector_frames
_bridge     — optional host-side IFC bridge (ifckit installs it)
"""

from geometrykit.geometry.biarc import fit_biarcs, solve_biarc
from geometrykit.geometry.curve import Curve, CurveReport
from geometrykit.geometry.field import (
    Field,
    FieldPoint,
    crossing_grid,
    divide_trim_edge,
    orient_division,
    trim_grid,
)
from geometrykit.geometry.frames import (
    FrameField,
    fixed_ref_frames,
    transport_frames,
    upvector_frames,
)
from geometrykit.geometry.intersection import Intersection
from geometrykit.geometry.panel import FieldReport, cells_report, field_report
from geometrykit.geometry.path import Path, PathReport, assemble_path
from geometrykit.geometry.patterns import (
    DiamondPattern,
    HexPattern,
    anglebox_cells,
    brick_cells,
    diamond_cells,
    hankin_cells,
    hex_cells,
    quad_cells,
    triangle_cells,
    truchet_cells,
    wave_cells,
)
from geometrykit.geometry.primitives import (
    Arc,
    ArcReport,
    Line,
    LineReport,
    Plane,
    PlaneReport,
    Polyline,
    Vec,
    VecReport,
    _polygon_normal,
    _signed_area,
)
from geometrykit.geometry.spiral import Spiral
from geometrykit.geometry.subdivision import catmull_clark, extract_patches, write_obj
from geometrykit.geometry.surface import Surface, SurfaceReport
from geometrykit.geometry.transform import Transform

__all__ = [
    # primitives
    "Vec",
    "Plane",
    "Line",
    "Arc",
    "Polyline",
    "_polygon_normal",
    "_signed_area",
    # reports
    "VecReport",
    "PlaneReport",
    "LineReport",
    "ArcReport",
    "PathReport",
    "CurveReport",
    "SurfaceReport",
    # path
    "Path",
    "assemble_path",
    # curve
    "Curve",
    "solve_biarc",
    "fit_biarcs",
    # spiral
    "Spiral",
    # surface
    "Surface",
    "Intersection",
    # field
    "Field",
    "FieldPoint",
    "trim_grid",
    "crossing_grid",
    "divide_trim_edge",
    "orient_division",
    # patterns
    "quad_cells",
    "DiamondPattern",
    "diamond_cells",
    "HexPattern",
    "hex_cells",
    "wave_cells",
    "anglebox_cells",
    "brick_cells",
    "truchet_cells",
    "hankin_cells",
    "triangle_cells",
    # panel
    "FieldReport",
    "cells_report",
    "field_report",
    # subdivision
    "catmull_clark",
    "extract_patches",
    "write_obj",
    # frames
    "FrameField",
    "transport_frames",
    "fixed_ref_frames",
    "upvector_frames",
    # transform
    "Transform",
]
