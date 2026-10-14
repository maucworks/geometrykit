# This file was generated with the assistance of an AI coding tool.
"""The spline cage lifted from the configurator corpus (plan M-cage)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from geometrykit.cage import CoordinateSystem, Corner
from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _load_slab(name: str = "flowing-roof") -> dict:
    data = json.loads((FIXTURES / "corpus" / f"{name}.json").read_text())
    slabs = [e for e in data["elements"] if "footprint" in e["body"]]
    assert slabs, f"{name}: no footprint element in corpus"
    return slabs[0]["body"]


def _plane_normal(body: dict) -> tuple:
    """Recorded plane x×y cross, normalized (the slab plane is axis-aligned here)."""
    plane = body["plane"]
    x = (plane["x_axis"]["x"], plane["x_axis"]["y"], plane["x_axis"]["z"])
    y = (plane["y_axis"]["x"], plane["y_axis"]["y"], plane["y_axis"]["z"])
    cross = (
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0],
    )
    ln = math.sqrt(sum(c * c for c in cross))
    return (cross[0] / ln, cross[1] / ln, cross[2] / ln)


def cage_from_slab(body: dict):
    """Build the spline cage from the flowing-roof slab footprint.

    - corners: **rebased datums** — origin at each footprint corner (the
      recorded slab plane origin lives 0.2 below the corners: thickness
      offset; the cage must sit on the corners), normal inherited from the
      recorded plane (= +z, validated below);
    - curves: the footprint's closed polyline as degree-1 kernel Curves
      (segment i leaves corner i, arrives at corner (i+1) mod n);
    - records: per corner two terminal angle records — `leave-{i}` for the
      outward edge direction, `arrive-{i}` for the incoming edge direction
      (measured in the frame's XY polar coordinates).
    """
    nrm = _plane_normal(body)
    pts = [Vec(*p) for p in body["footprint"]]
    n = len(pts)

    curves = []
    for i in range(n):
        curves.append(
            Curve(
                control_points=[pts[i], pts[(i + 1) % n]],
                knots=[0.0, 1.0],
                multiplicities=[2, 2],
                degree=1,
            )
        )

    xref = pts[1] - pts[0]  # first edge direction as the frame's +x reference
    corners = []
    for i in range(n):
        cs = CoordinateSystem.from_plane(pts[i], nrm, (xref.x, xref.y, xref.z), id=f"cs-{i}")
        out_dir = pts[(i + 1) % n] - pts[i]
        in_dir = pts[i] - pts[(i - 1) % n]
        ta = cs.apply_direction((out_dir.x, out_dir.y, out_dir.z))
        tb = cs.apply_direction((in_dir.x, in_dir.y, in_dir.z))
        angles = {
            f"leave-{i}": math.atan2(ta[1], ta[0]),
            f"arrive-{i}": math.atan2(tb[1], tb[0]),
        }
        corners.append(Corner(cs, angles, id=f"corner-{i}"))
    return corners, curves, pts


class TestCageExample:
    def test_cage_builds_and_validates(self):
        body = _load_slab()
        corners, curves, pts = cage_from_slab(body)
        assert len(corners) == 4 and len(curves) == 4

        # corners sit exactly on their own footprint corners (rebased datums)
        for i, corner in enumerate(corners):
            assert all(abs(a - b) < TOL for a, b in zip(corner.coordinate_system.position, (pts[i].x, pts[i].y, pts[i].z)))

        # terminal records validate against the measured curve tangents:
        # recorded leave-angle ↔ curve.tangent_at(0); the corner's arrive-
        # record is the *arrival* record for the shared seam: the arrival
        # curve's end tangent equals the incoming edge's departure direction.
        for i, (corner, curve) in enumerate(zip(corners, curves)):
            t_a = curve.tangent_at(0.0)
            t_e = corner.tangent_for(f"leave-{i}")
            assert all(abs(getattr(t_a, k) - x) < TOL for k, x in zip("xyz", t_e)), (i, t_a, t_e)
            next_corner = corners[(i + 1) % len(corners)]
            t_a = curve.tangent_at(1.0)
            t_e = next_corner.tangent_for(f"arrive-{(i + 1) % len(corners)}")
            assert all(abs(getattr(t_a, k) - x) < TOL for k, x in zip("xyz", t_e)), (i, t_a, t_e)

    def test_in_plane_tangents(self):
        """All terminal tangents have zero z-component (the XY planarity rule)."""
        body = _load_slab()
        corners, curves, _ = cage_from_slab(body)
        for i, (corner, curve) in enumerate(zip(corners, curves)):
            for t in (curve.tangent_at(0.0), curve.tangent_at(1.0)):
                assert abs(t.z) < 1e-12, (i, t)

    def test_wedge_angles_non_degenerate(self):
        body = _load_slab()
        corners, _, _ = cage_from_slab(body)
        for corner in corners:
            corner.validate_terminals()

    def test_twist_rule_theta_minus_90(self):
        """Ribbon twist at a corner: τ = T × n — an in-frame θ − 90° rotation."""
        body = _load_slab()
        corners, _, _ = cage_from_slab(body)
        corner = corners[0]
        theta = corner.angles["leave-0"]
        tangent = corner.tangent_for("leave-0")
        t_vec = tuple(tangent)
        normal = corner.coordinate_system.apply_direction((0.0, 0.0, 1.0))
        tau_cross = (
            t_vec[1] * normal[2] - t_vec[2] * normal[1],
            t_vec[2] * normal[0] - t_vec[0] * normal[2],
            t_vec[0] * normal[1] - t_vec[1] * normal[0],
        )
        local = (
            math.cos(theta - math.pi / 2),
            math.sin(theta - math.pi / 2),
            0.0,
        )
        tau_frame = corner.coordinate_system.apply_direction(local)
        for a, b in zip(tau_cross, tau_frame):
            assert abs(a - b) < 1e-9, (tau_cross, tau_frame)


TOL = 1e-9
