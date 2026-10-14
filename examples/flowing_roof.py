# This file was generated with the assistance of an AI coding tool.
"""flowing-roof — the standalone geometry stress case (ported example).

The pipeline, stage by stage (each stage maps to kernel milestones):

1. seven curves from tangents (G1 end tangents) — `Curve.from_tangents`,
   pure kernel today;
2. six corners where those curves meet, each with its associated
   ribbons — `Surface.ribbon`, OCCT-gated until M6;
3. two G1-constrained patches over the shared `top` curve — M7/M8
   (OCCT twin until then); the ridge ribbon is shared, so it is symmetric
   (spine in the middle) and the constraint is read on the spine;
4. an array of planes intersected with both patches — plane×surface
   sectioning (M5), OCCT-gated;
5. the intersection curves rationalized into bi-arc chains —
   `geometry.biarc` is the pure module behind it (the beam-path
   contract: sections become Line/Arc chains feeding element paths).

Importable and runnable without a server, an IFC file or a viewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from geometrykit.geometry.curve import Curve
from geometrykit.geometry.intersection import Intersection
from geometrykit.geometry.primitives import Plane, Vec
from geometrykit.geometry.surface import Surface


@dataclass
class _Miters:
    left_bottom: Plane
    right_bottom: Plane
    left_top: Plane
    right_top: Plane
    left: Plane
    right: Plane
    top: Plane
    bottom: Plane


def _make_miters(width: float, height: float, ipe_height: float) -> _Miters:
    return _Miters(
        left_bottom=Plane(Vec(0, 0, 0), Vec(1, 1, 0), Vec(0, 0, 1)),
        right_bottom=Plane(Vec(width, 0, 0), Vec(1, -1, 0), Vec(0, 0, 1)),
        left_top=Plane(Vec(0, height, 0), Vec(-1, 1, 0), Vec(0, 0, 1)),
        right_top=Plane(Vec(width, height, 0), Vec(-1, -1, 0), Vec(0, 0, 1)),
        left=Plane(Vec(ipe_height / 2, 0, 0), Vec(0, -1, 0), Vec(0, 0, 1)),
        right=Plane(Vec(width - ipe_height / 2, 0, 0), Vec(0, 1, 0), Vec(0, 0, 1)),
        top=Plane(Vec(0, height - ipe_height / 2, 0), Vec(-1, 0, 0), Vec(0, 0, 1)),
        bottom=Plane(Vec(0, ipe_height / 2, 0), Vec(1, 0, 0), Vec(0, 0, 1)),
    )


@dataclass
class RoofGeometry:
    """All constructed geometry (inputs to element assembly)."""

    width: float = 8
    height: float = 12
    ipe_height: float = 0.2
    slab_thickness: float = 0.2
    miters: _Miters | None = None
    bottom: Curve | None = None
    right: Curve | None = None
    top: Curve | None = None
    left: Curve | None = None
    right2: Curve | None = None
    top2: Curve | None = None
    left2: Curve | None = None
    ribbons: dict = field(default_factory=dict)
    surf: Surface | None = None
    surf2: Surface | None = None


def build_curves(params: dict) -> RoofGeometry:
    """Stage 1 (pure): miter planes + the seven interpolated edge curves."""
    g = RoofGeometry()
    height_sw = params["height_sw"]
    height_se = params["height_se"]
    height_nw = params["height_nw"]
    height_ne = params["height_ne"]
    height_nnw = params["height_nnw"]
    height_nne = params["height_nne"]
    # the twist lives in build_ribbons; the curves only need the heights
    width, height = g.width, g.height

    g.miters = _make_miters(width, height, g.ipe_height)

    g.bottom = Curve.from_tangents(
        start=Vec(0, 0, height_sw),
        tan_start=Vec(1, 0, 0),
        end=Vec(width, 0, height_se),
        tan_end=Vec(1, 0, 0),
        scale=3.0,
    )
    g.right = Curve.from_tangents(
        start=Vec(width, 0, height_se),
        tan_start=Vec(0, 1, 0),
        end=Vec(width, 6, height_ne),
        tan_end=Vec(0, 1, 0),
        scale=3.0,
    )
    g.top = Curve.from_tangents(
        start=Vec(0, height / 2, height_nw),
        tan_start=Vec(1, 0, 0),
        end=Vec(width, height / 2, height_ne),
        tan_end=Vec(1, 0, 0),
        scale=3.0,
    )
    g.left = Curve.from_tangents(
        start=Vec(0, 6, height_nw),
        tan_start=Vec(0, -1, 0),
        end=Vec(0, 0, height_sw),
        tan_end=Vec(0, -1, 0),
        scale=3.0,
    )
    g.right2 = Curve.from_tangents(
        start=Vec(width, 12, height_nne),
        end=Vec(width, 6, height_ne),
        tan_start=Vec(0, -1, 0),
        tan_end=Vec(0, -1, 0),
        scale=3.0,
    )
    g.top2 = Curve.from_tangents(
        start=Vec(0, height, height_nnw),
        end=Vec(width, height, height_nne),
        tan_start=Vec(1, 0, 0),
        tan_end=Vec(1, 0, 0),
        scale=3.0,
    )
    g.left2 = Curve.from_tangents(
        start=Vec(0, 6, height_nw),
        end=Vec(0, 12, height_nnw),
        tan_start=Vec(0, 1, 0),
        tan_end=Vec(0, 1, 0),
        scale=3.0,
    )
    return g


def build_ribbons(g: RoofGeometry, twist: int = 25) -> RoofGeometry:
    """Stage 2 (M6, OCCT twin today): the seven ribbons.

    Every ribbon is **symmetric**: its spine curve runs down the middle of the
    strip, so the tangent plane is read on the spine rather than on one side of
    it, and the same support serves a patch on either side of the curve. That
    matters at the ridge, where one ribbon constrains two patches.
    """
    g.ribbons = {
        "top": Surface.ribbon(
            g.top, angles_deg=[180, 180 - twist, 180], width=1, symmetric=True
        ),
        "bot": Surface.ribbon(g.bottom, angle_deg=0, width=1, symmetric=True),
        "left": Surface.ribbon(g.left, angle_deg=0, width=1, symmetric=True),
        "right": Surface.ribbon(g.right, angle_deg=0, width=1, symmetric=True),
        "top2": Surface.ribbon(g.top2, angle_deg=180, width=1, symmetric=True),
        "right2": Surface.ribbon(g.right2, angle_deg=180, width=1, symmetric=True),
        "left2": Surface.ribbon(g.left2, angle_deg=180, width=1, symmetric=True),
    }
    return g


def build_patches(g: RoofGeometry, tol: float = 0.0001) -> RoofGeometry:
    """Stage 3 (M7/M8, OCCT twin today): the two G1-constrained patches."""

    g.surf = Surface.patch(
        curves=[g.bottom, g.right, g.top, g.left],
        constraints=["G1", "G1", "G1", "G1"],
        supports=[
            g.ribbons["bot"],
            g.ribbons["right"],
            g.ribbons["top"],
            g.ribbons["left"],
        ],
        tol=0.0001,
    )
    g.surf2 = Surface.patch(
        curves=[g.top, g.right2, g.top2, g.left2],
        constraints=["G1", "G1", "G1", "G1"],
        supports=[
            g.ribbons["top"],
            g.ribbons["right2"],
            g.ribbons["top2"],
            g.ribbons["left2"],
        ],
        tol=0.0001,
    )
    return g


def sweep_curves(
    surf: Surface,
    surf2: Surface,
    plane_normal: Vec,
    num: int,
    span: float,
) -> list:
    """Intersect both patches with sweep planes: (plane, curves) per plane.

    One section per plane per patch; keep plane counts low on interactive
    sliders (each plane costs ~2 section builds).
    """
    sets: list = []
    if num == 0:
        return sets
    step = span / num
    for i in range(num):
        p = Plane.from_origin_and_normal(Vec(0, -(span / 2) + step * i, 0), plane_normal)
        curves = list(Intersection.of(surf, p).curves)
        curves.extend(Intersection.of(surf2, p).curves)
        sets.append((p, Curve.assemble(curves, tol=0.1)))
    return sets


def build_roof(params: dict) -> RoofGeometry:
    """Full chain: curves → ribbons → patches (OCCT twin for stages 2–3)."""
    g = build_curves(params)
    build_ribbons(g, twist=params.get("twist", 25))
    build_patches(g)
    return g
