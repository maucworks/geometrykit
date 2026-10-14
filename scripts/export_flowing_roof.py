# This file was generated with the assistance of an AI coding tool.
"""Export the flowing-roof patches (P-1 kernel + OCCT twin) to OBJ/STL/STEP.

Run in an env with pythonocc (the ribbon normals and the twin):

    /opt/miniconda3/envs/a320-ifc-configurator/bin/python \
        scripts/export_flowing_roof.py

Output goes to ``wip/flowing-roof-eval/`` (gitignored scratch: generated
results are inspected, not committed).

All three writers are occ-free; only the constraint normals (OCCT ribbon
surfaces) and the twin comparison need pythonocc.
"""

from __future__ import annotations

import importlib.util
import math
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


def ribbon_field(edge, curve, ribbon, oriented, params_u, params_v):
    """A patch-parameter -> ribbon-normal callable for one edge.

    Maps each patch station onto the edge curve's own parameter (a drawn
    direction may run opposite to the patch's sampled direction) and reads
    the ribbon's normal on the long edge it shares with that curve.
    """
    patch_params = params_u if edge in ("south", "north") else params_v
    curve_params = patch_np.edge_curve_params(oriented, edge, patch_params)

    def at(t):
        k = min(range(len(curve_params)), key=lambda m: abs(curve_params[m] - t))
        t_curve = curve_params[k]
        base = curve.point_at(t_curve)
        best = None
        for v_edge in (0.0, 1.0):
            p = ribbon.point_at(t_curve, v_edge)
            d = abs(p - base)
            if best is None or d < best[0]:
                best = (d, v_edge)
        dist, v_edge = best
        if dist > 1e-4:
            raise SystemExit(f"{edge}: ribbon misses the curve at t={t_curve} ({dist:.2e})")
        n = ribbon.normal_at(t_curve, v_edge)
        ln = math.sqrt(n.x**2 + n.y**2 + n.z**2) or 1.0
        return (n.x / ln, n.y / ln, n.z / ln)

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

    curves = [g.bottom, g.right, g.top, g.left]
    params_u, params_v, oriented = patch_np.coons_parametrization(curves, nu=16, nv=16)
    fields = {
        edge: ribbon_field(edge, curve, g.ribbons[ribbon_name], oriented, params_u, params_v)
        for edge, ribbon_name in EDGES
        for curve in [curves[[e for e, _r in EDGES].index(edge)]]
    }
    p1_surf, _corners, stations = patch_g1.constrained_coons_patch(
        curves, fields, nu=16, nv=16
    )

    worst = 0.0
    for i in range(1, 16):
        for j in range(1, 16):
            u, v = i / 16.0, j / 16.0
            ours = p1_surf.point_at(u, v)
            twin, _uv = g.surf.project(ours)
            worst = max(worst, abs(ours - twin))
    print(f"P-1 kernel patch built; max deviation from the OCCT twin: {worst:.4f}")
    for edge in patch_g1.EDGE_NAMES:
        params = stations[0] if edge in ("south", "north") else stations[1]
        print(f"  G1 residual {edge:>5}: {edge_residual(p1_surf, edge, params, fields[edge]):.3e}")

    # One file, three named groups, no OCCT twin: the kernel patch, the four
    # constraint ribbons it was solved against, and the seven edge curves the
    # network is made of. A zebra or curvature analysis across the patch to
    # ribbon seams is the visual counterpart of the G1 residuals above.
    surfaces = [p1_surf] + [g.ribbons[name] for _edge, name in EDGES]
    surface_labels = ["PatchKernelG1"] + [f"Ribbon_{edge}" for edge, _name in EDGES]
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