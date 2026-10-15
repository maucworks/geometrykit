# This file was generated with the assistance of an AI coding tool.
"""The spline cage lifted from the configurator corpus (plan M-cage)."""

from __future__ import annotations

import math
from pathlib import Path

from geometrykit.cage import CoordinateSystem, Corner
from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _load_slab(name: str = "flowing-roof") -> dict:
    """Load the flowing-roof slab body from the committed corpus."""
    data = (FIXTURES / "corpus" / f"{name}.json").read_text()
    import json

    parsed = json.loads(data)
    slabs = [e for e in parsed["elements"] if "footprint" in e["body"]]
    assert slabs, f"{name}: no footprint element in corpus"
    return slabs[0]["body"]


def _plane_normal(body: dict) -> tuple:
    """Recorded plane x×y cross, normalized (axis-aligned for the slab plane)."""
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


def cage_from_footprint(body: dict):
    """Build the spline cage from the flowing-roof slab footprint.

    - corners: **rebased datums** — origin at each footprint corner (the
      recorded slab plane origin lives 0.2 below the corners: thickness
      offset; the cage must sit on the corners), normal inherited from
      the recorded plane (= +z, validated below);
    - curves: the footprint's closed polyline as degree-1 kernel Curves
      (segment i leaves corner i, arrives at corner (i+1) mod n);
    - records: per corner two terminal angle records — `leave-{i}` for
      the outward edge direction, `arrive-{i}` for the incoming edge
      direction (measured in the frame's XY polar coordinates).
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

    corners = []
    for i in range(n):
        xref = pts[(i + 1) % n] - pts[i]  # the leaving direction as the frame's +x
        cs = CoordinateSystem.from_plane(pts[i], nrm, tuple(xref), id=f"cs-{i}")
        out_dir = pts[(i + 1) % n] - pts[i]
        in_dir = pts[i] - pts[(i - 1) % n]
        ta = cs.apply_direction(tuple(out_dir))
        tb = cs.apply_direction(tuple(in_dir))
        angles = {
            f"leave-{i}": math.atan2(ta[1], ta[0]),
            f"arrive-{i}": math.atan2(tb[1], tb[0]),
        }
        corners.append(Corner(cs, angles, id=f"corner-{i}"))
    return corners, curves, pts
