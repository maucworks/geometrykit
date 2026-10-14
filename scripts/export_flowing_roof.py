# This file was generated with the assistance of an AI coding tool.
"""Export the flowing-roof patches (P-1 kernel + OCCT twin) to OBJ/STL/STEP.

Run in an env with pythonocc (the ribbon normals and the twin):

    conda run -n geometrykit-oracle python scripts/export_flowing_roof.py

Output goes to ``wip/flowing-roof-eval/`` (gitignored scratch: generated
results are inspected, not committed).

All three writers are occ-free; only the constraint normals (OCCT ribbon
surfaces) and the twin comparison need pythonocc.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

_EXAMPLE = ROOT / "examples" / "flowing_roof.py"
_spec = importlib.util.spec_from_file_location("flowing_roof", _EXAMPLE)
fr = importlib.util.module_from_spec(_spec)
sys.modules["flowing_roof"] = fr
_spec.loader.exec_module(fr)

from geometrykit.export import write_step_geometry  # noqa: E402
from geometrykit.geometry.primitives import Vec  # noqa: E402
from geometrykit.runtimes._npu import patch_g1, patch_np  # noqa: E402

# Tangency strength: how stiffly the patches leave their edges, as a multiple
# of the support ribbons' own slope (1.0 = the ribbon's rate, None = the
# solver's minimum-norm leftover, which is softer). Dial this and re-run.
TANGENCY = 0.75  # the variant the user's eye accepted over 1.0; the
# band/core curvature metric disagrees (2.1x vs 1.3x) -- the metric lost
# this one and the eye won

# Also match the ribbons' *curvature* across each edge (G2): the patches stop
# bending away from the ribbon's plane as they leave the edge. The ribbons are
# flat strips, so this drives their cross-curvature to zero.
G2 = True

# Station parameterisation: "chord" or "arclength". Arc-length stations pair
# with uniform knots and gave the more even iso-parametric spacing on this
# network (ratio 1.36 vs 1.75), with cross-curvature 2x lower.
PARAMETRIZATION = "arclength"

# Loose fill: stations of enlargement per side (0 = the classical fill). With
# loose > 0 the plate is solved over a grid enlarged by that many stations on
# every side -- the through-the-edge construction OCCT's plate uses -- and the
# window returns: the original edges stay exact interior lines, the plate gets
# a margin to spend bending on, and the belt is trimmed away. Measured on this
# network at tangency 0.75 (probe density): cross-curvature band/core 3.44x
# against 2.35x tight, dev from twin 0.038 vs 0.029 -- the metric says the eye
# should judge this one, as it did for the tangency sweep.
LOOSE = 0

# Solve the interior as an energy-minimising fill instead of taking the Coons
# blend: the boundary ring still interpolates the boundary curves exactly, and
# the interior is the thin-plate solution through it (simply supported). That
# is what gives the fill its even curvature instead of a flat middle whose
# bending is pushed onto the edges. Measured on this network: cross-direction
# curvature 2.4x lower than the blend before the continuity solve, and the
# whole M0+M2 configuration cuts the cross-direction peak 8x against the
# blend in chord mode.
FILL = True

# the mixed-height network: the real non-planar G1 stress case
PARAMS = {
    "height_sw": 2.4,
    "height_se": 2.8,
    "height_nw": 3.2,
    "height_ne": 2.9,
    "height_nnw": 3.4,
    "height_nne": 3.0,
    "twist": 25,
}

# first patch: its four edges and the ribbon riding each of them
EDGES = (
    ("south", "bot"),
    ("east", "right"),
    ("north", "top"),
    ("west", "left"),
)


def spine_field(edge, spine, support, oriented, params_u, params_v, towards):
    """The constraint field of *edge*, read on *support*'s spine.

    The spine is located on the support by projection, so it does not matter
    whether the ribbon is one-sided or symmetric, nor how either is
    parameterised. The spec carries the support and its spine as well, which
    is what a tangency strength is scaled against.
    """
    patch_params = params_u if edge in ("south", "north") else params_v
    curve_params = patch_np.edge_curve_params(oriented, edge, patch_params)
    spec = patch_g1.support_constraint(support, spine, towards=towards)

    def _map(field):
        def at(t):
            return field(min(curve_params, key=lambda c: abs(c - t)))

        return at

    return {
        "normal": _map(spec["normal"]),
        "slope": _map(spec["slope"]),
        "support": support,
        "spine": spine,
        "curve_params": curve_params,
    }


def curvature_gap(surface, field, edge, t) -> float:
    """Patch cross-curvature minus the support's, at one boundary station."""
    step = 1e-5
    if edge in ("south", "north"):
        v0 = 0.0 if edge == "south" else 1.0
        delta = step if edge == "south" else -step
        first = surface.tangents_at(t, v0)[1]
        later = surface.tangents_at(t, v0 + delta)[1]
        normal = surface.normal_at(t, v0)
    else:
        u0 = 0.0 if edge == "west" else 1.0
        delta = step if edge == "west" else -step
        first = surface.tangents_at(u0, t)[0]
        later = surface.tangents_at(u0 + delta, t)[0]
        normal = surface.normal_at(u0, t)
    second = (
        (later.x - first.x) / delta,
        (later.y - first.y) / delta,
        (later.z - first.z) / delta,
    )
    speed = first.x**2 + first.y**2 + first.z**2
    patch_kappa = (
        second[0] * normal.x + second[1] * normal.y + second[2] * normal.z
    ) / speed
    spine = field["spine"]
    curve_params = field["curve_params"]
    support_kappa = patch_g1.support_curvature_field(field["support"], spine)(
        min(curve_params, key=lambda c: abs(c - t))
    )
    return patch_kappa - support_kappa


def edge_residual(surface, edge, params, field) -> float:
    """Worst |cross-derivative . constraint normal| outside the corner band."""
    worst = 0.0
    for t in params[2:-2]:
        if edge == "south":
            d = surface.tangents_at(t, 0.0)[1]
        elif edge == "north":
            d = surface.tangents_at(t, 1.0)[1]
        elif edge == "west":
            d = surface.tangents_at(0.0, t)[0]
        else:
            d = surface.tangents_at(1.0, t)[0]
        n = field(t)
        worst = max(worst, abs(d.x * n[0] + d.y * n[1] + d.z * n[2]))
    return worst


def main() -> None:
    g = fr.build_curves(PARAMS)
    fr.build_ribbons(g, twist=PARAMS["twist"])
    fr.build_patches(g)

    out_dir = ROOT / "wip" / "flowing-roof-eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Two patches over the shared ridge curve, exactly like the example:
    # front = bottom/right/top/left, back = top/right2/top2/left2.
    networks = (
        ("Front", ["bottom", "right", "top", "left"], ["bot", "right", "top", "left"]),
        ("Back", ["top", "right2", "top2", "left2"], ["top", "right2", "top2", "left2"]),
    )
    patches = []
    for name, spine_names, support_names in networks:
        spines = [getattr(g, n) for n in spine_names]
        supports = [g.ribbons[n] for n in support_names]
        params_u, params_v, oriented = patch_np.coons_parametrization(
            spines, nu=16, nv=16, parametrization=PARAMETRIZATION
        )
        towards = Vec(
            sum(c.point_at(0.5).x for c in spines) / 4,
            sum(c.point_at(0.5).y for c in spines) / 4,
            sum(c.point_at(0.5).z for c in spines) / 4,
        )
        fields = {
            edge: spine_field(edge, spine, support, oriented, params_u, params_v, towards)
            for edge, spine, support in zip(patch_g1.EDGE_NAMES, spines, supports)
        }
        surface, _corners, stations = patch_g1.constrained_coons_patch(
            spines, fields, nu=16, nv=16, tangency=TANGENCY, g2=G2, fill=FILL,
            loose=LOOSE, parametrization=PARAMETRIZATION
        )
        patches.append((name, surface))
        label = "solver default" if TANGENCY is None else f"tangency {TANGENCY}"
        label += ", G2" if G2 else ", G1 only"
        label += ", plate fill" if FILL else ", Coons blend"
        print(f"{name} patch built from the kernel ({label})")
        for edge in patch_g1.EDGE_NAMES:
            params = stations[0] if edge in ("south", "north") else stations[1]
            worst = edge_residual(surface, edge, params, fields[edge]["normal"])
            slopes = []
            for t in params[2:-2]:
                if edge in ("south", "north"):
                    cross = surface.tangents_at(t, 0.0)[1] if edge == "south" else (
                        surface.tangents_at(t, 1.0)[1]
                    )
                else:
                    cross = surface.tangents_at(0.0, t)[0] if edge == "west" else (
                        surface.tangents_at(1.0, t)[0]
                    )
                slopes.append(abs(cross) / max(patch_g1._cross_extent(surface, edge, t), 1e-9))
            gaps = []
            for t in params[2:-2]:
                gaps.append(abs(curvature_gap(surface, fields[edge], edge, t)))
            print(
                f"  {edge:>5}: G1 {worst:.3e}   boundary slope "
                f"{sum(slopes) / len(slopes):.3f}   curvature gap "
                f"{sum(gaps) / len(gaps):.2e}"
            )

    # One file, three named groups, no OCCT twin: the kernel patch, the four
    # constraint ribbons it was solved against, and the seven edge curves the
    # network is made of. A zebra or curvature analysis across the patch to
    # ribbon seams is the visual counterpart of the G1 residuals above.
    surfaces = [surface for _name, surface in patches]
    surface_labels = [f"Patch{name}" for name, _surface in patches]
    ribbon_labels = (
        ("bot", "Ribbon_front_eave"),
        ("right", "Ribbon_front_east"),
        ("top", "Ribbon_ridge_shared"),
        ("left", "Ribbon_front_west"),
        ("right2", "Ribbon_back_east"),
        ("top2", "Ribbon_back_eave"),
        ("left2", "Ribbon_back_west"),
    )
    for ribbon_name, label in ribbon_labels:
        surfaces.append(g.ribbons[ribbon_name])
        surface_labels.append(label)
    edges = [
        ("Edge_bottom", g.bottom),
        ("Edge_right", g.right),
        ("Edge_top_ridge", g.top),
        ("Edge_left", g.left),
        ("Edge_right_back", g.right2),
        ("Edge_top_back_eave", g.top2),
        ("Edge_left_back", g.left2),
    ]
    suffix = "" if LOOSE <= 0 else f"-loose{LOOSE}"
    step_path = out_dir / f"flowing-roof{suffix}.step"
    write_step_geometry(
        str(step_path),
        surfaces=surfaces,
        surface_labels=surface_labels,
        curves=[curve for _label, curve in edges],
        curve_labels=[label for label, _curve in edges],
    )
    print(f"STEP -> {step_path}")
    for label, curve in edges:
        a, b = curve.point_at(0.0), curve.point_at(1.0)
        print(
            f"  {label:>20}: ({a.x:5.2f},{a.y:5.2f},{a.z:5.2f}) -> "
            f"({b.x:5.2f},{b.y:5.2f},{b.z:5.2f})"
        )


if __name__ == "__main__":
    main()