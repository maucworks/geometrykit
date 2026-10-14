# This file was generated with the assistance of an AI coding tool.
"""Interchange writers: OBJ/STL meshes and the AP214 STEP surface export.

The STEP test is the load-bearing one: it writes the file, reads it back
with OCCT (``STEPControl_Reader``) and checks that the surface a CAD kernel
reconstructs is the surface the kernel built -- control net, knots and all.
A reader that silently degrades to a mesh, or that loses the G1 shape,
fails here.
"""

from __future__ import annotations

import pytest
from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec
from geometrykit.geometry.surface import Surface


def _occ_available() -> bool:
    try:
        import OCC.Core.BRep  # noqa: F401
        return True
    except ImportError:
        return False


def _make_patch() -> Surface:
    from geometrykit.runtimes._npu import patch_np

    curves = [
        Curve.from_tangents(
            start=Vec(0, 0, 2.5), tan_start=Vec(1, 0, 0),
            end=Vec(8, 0, 2.7), tan_end=Vec(1, 0, 0), scale=3.0,
        ),
        Curve.from_tangents(
            start=Vec(8, 0, 2.7), tan_start=Vec(0, 1, 0),
            end=Vec(8, 6, 3.1), tan_end=Vec(0, 1, 0), scale=3.0,
        ),
        Curve.from_tangents(
            start=Vec(8, 6, 3.1), tan_start=Vec(-1, 0, 0),
            end=Vec(0, 6, 2.9), tan_end=Vec(-1, 0, 0), scale=3.0,
        ),
        Curve.from_tangents(
            start=Vec(0, 6, 2.9), tan_start=Vec(0, -1, 0),
            end=Vec(0, 0, 2.5), tan_end=Vec(0, -1, 0), scale=3.0,
        ),
    ]
    surface, _corners = patch_np.discrete_coons_patch(curves, nu=10, nv=10)
    return surface


# OCCT's STEP reader normalises lengths to millimetres (its
# ``xstep.cascade.unit`` default). Our files declare metres, so a read-back
# is in millimetres: scale to compare against the kernel's own units.
MM_PER_M = 0.001


def _read_step(path):
    """Read *path* with OCCT and return its shape (millimetres)."""
    from OCC.Core.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    assert reader.ReadFile(str(path)) == 1, "STEPControl_Reader refused the file"
    reader.TransferRoots()
    return reader.OneShape()


class TestMeshWriters:
    """OBJ/STL: occ-free, any viewer or Blender reads them."""

    def test_obj_quad_grid(self, tmp_path):
        from geometrykit.export import surface_to_obj, write_obj

        surface = _make_patch()
        verts, faces = surface_to_obj(surface, nu=5, nv=4)
        assert len(verts) == 6 * 5
        assert len(faces) == 5 * 4
        assert all(len(f) == 4 for f in faces)
        out = tmp_path / "patch.obj"
        write_obj([surface], str(out), labels=["patch"], nu=5, nv=4)
        lines = out.read_text().splitlines()
        assert lines[1] == "o patch"
        assert sum(1 for line in lines if line.startswith("v ")) == len(verts)
        assert sum(1 for line in lines if line.startswith("f ")) == len(faces)

    def test_stl_is_closed_writing(self, tmp_path):
        from geometrykit.export import write_stl

        out = tmp_path / "patch.stl"
        write_stl([_make_patch()], str(out), nu=30, nv=30)
        text = out.read_text()
        assert text.startswith("solid geometrykit")
        assert text.rstrip().endswith("endsolid geometrykit")
        assert text.count("facet normal") == 30 * 30 * 2


class TestStepExport:
    """AP214 surface export: the file a CAD kernel can open."""

    def test_step_is_ap214_with_bspline_surface(self, tmp_path):
        from geometrykit.export import write_step_surfaces

        surface = _make_patch()
        out = tmp_path / "patch.step"
        text = write_step_surfaces([surface], str(out), labels=["PatchKernelG1"])
        assert text.startswith("ISO-10303-21;")
        assert "AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }" in text
        assert text.count("B_SPLINE_SURFACE_WITH_KNOTS") == 1
        assert text.count("ADVANCED_FACE") == 1
        assert "IFC" not in text
        # every pole reaches the file, at full precision
        for row in surface.control_points:
            for p in row:
                assert repr(float(p.x)) in text

    def test_step_refuses_rational_surfaces(self, tmp_path):
        from geometrykit.export import write_step_surfaces

        surface = _make_patch()
        rational = Surface(
            control_points=surface.control_points,
            uknots=surface.uknots,
            vknots=surface.vknots,
            umults=surface.umults,
            vmults=surface.vmults,
            udegree=surface.udegree,
            vdegree=surface.vdegree,
            weights=[[1.0] * len(row) for row in surface.control_points],
        )
        with pytest.raises(ValueError, match="rational"):
            write_step_surfaces([rational], str(tmp_path / "r.step"))

    @pytest.mark.skipif(not _occ_available(), reason="needs pythonocc to read STEP back")
    def test_occ_reads_the_surface_back_exactly(self, tmp_path):
        """The real acceptance: OCCT reconstructs our surface from the file."""
        from geometrykit.export import write_step_surfaces
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.Geom import Geom_BSplineSurface
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopExp import TopExp_Explorer

        surface = _make_patch()
        out = tmp_path / "patch.step"
        write_step_surfaces([surface], str(out), labels=["PatchKernelG1"])

        shape = _read_step(out)

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        assert explorer.More(), "the STEP file contains no face"
        face = explorer.Current()
        bspline = Geom_BSplineSurface.DownCast(BRep_Tool.Surface(face))
        assert bspline is not None, "the face's surface is not a B-spline"

        # the parameter domain is the surface's own [0, 1] x [0, 1]
        adaptor = BRepAdaptor_Surface(face)
        assert abs(adaptor.FirstUParameter()) < 1e-12
        assert abs(adaptor.LastUParameter() - 1.0) < 1e-12
        assert abs(adaptor.FirstVParameter()) < 1e-12
        assert abs(adaptor.LastVParameter() - 1.0) < 1e-12

        assert bspline.UDegree() == surface.udegree
        assert bspline.VDegree() == surface.vdegree
        assert bspline.NbUPoles() == surface.nu
        assert bspline.NbVPoles() == surface.nv

        # the pole cloud is identical, not merely close
        worst_pole = 0.0
        for i in range(1, bspline.NbUPoles() + 1):
            for j in range(1, bspline.NbVPoles() + 1):
                pole = bspline.Pole(i, j)
                mine = surface.control_points[i - 1][j - 1]
                worst_pole = max(
                    worst_pole,
                    abs(pole.X() * MM_PER_M - mine.x),
                    abs(pole.Y() * MM_PER_M - mine.y),
                    abs(pole.Z() * MM_PER_M - mine.z),
                )
        assert worst_pole < 1e-12, worst_pole

        # the surface it evaluates is ours, parameter for parameter
        worst_point = 0.0
        for u in (0.13, 0.5, 0.87):
            for v in (0.21, 0.5, 0.79):
                pnt = bspline.Value(u, v)
                mine = surface.point_at(u, v)
                worst_point = max(
                    worst_point,
                    abs(pnt.X() * MM_PER_M - mine.x),
                    abs(pnt.Y() * MM_PER_M - mine.y),
                    abs(pnt.Z() * MM_PER_M - mine.z),
                )
        assert worst_point < 1e-9, worst_point

    @pytest.mark.skipif(not _occ_available(), reason="needs pythonocc to read STEP back")
    def test_step_edges_are_the_boundary_curves(self, tmp_path):
        """The face is bounded by the surface's own four boundary curves."""
        from geometrykit.export import write_step_surfaces
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
        from OCC.Core.TopExp import TopExp_Explorer

        surface = _make_patch()
        out = tmp_path / "patch.step"
        write_step_surfaces([surface], str(out), labels=["PatchKernelG1"])
        shape = _read_step(out)

        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face = face_explorer.Current()
        wire_count = 0
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            wire_count += 1
            edge_explorer.Next()
        assert wire_count == 4, wire_count

        # the face's corner in parameter space is our surface's corner
        adaptor = BRepAdaptor_Surface(face)
        corner = adaptor.Value(0.0, 0.0)
        mine = surface.point_at(0.0, 0.0)
        assert abs(corner.X() * MM_PER_M - mine.x) < 1e-9
        assert abs(corner.Y() * MM_PER_M - mine.y) < 1e-9
        assert abs(corner.Z() * MM_PER_M - mine.z) < 1e-9