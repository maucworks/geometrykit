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
from geometrykit.runtimes._npu import patch_g1, patch_np  # noqa: E402

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


def spine_field(edge, spine, support, oriented, params_u, params_v):
    """The constraint normal of *edge*, read on *support*'s spine.

    The spine is located on the support by projection (see
    ``patch_g1.normal_field_from_support``), so it does not matter whether the
    ribbon is one-sided or symmetric, nor how either is parameterised.
    """
    patch_params = params_u if edge in ("south", "north") else params_v
    curve_params = patch_np.edge_curve_params(oriented, edge, patch_params)
    on_spine = patch_g1.normal_field_from_support(support, spine)

    def at(t):
        k = min(range(len(curve_params)), key=lambda m: abs(curve_params[m] - t))
        return on_spine(curve_params[k])

    return at


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
        params_u, params_v, oriented = patch_np.coons_parametrization(spines, nu=16, nv=16)
        fields = {
            edge: spine_field(edge, spine, support, oriented, params_u, params_v)
            for edge, spine, support in zip(patch_g1.EDGE_NAMES, spines, supports)
        }
        surface, _corners, stations = patch_g1.constrained_coons_patch(
            spines, fields, nu=16, nv=16
        )
        patches.append((name, surface))
        print(f"{name} patch built from the kernel")
        for edge in patch_g1.EDGE_NAMES:
            params = stations[0] if edge in ("south", "north") else stations[1]
            print(
                f"  G1 residual {edge:>5}: "
                f"{edge_residual(surface, edge, params, fields[edge]):.3e}"
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
    step_path = out_dir / "flowing-roof.step"
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