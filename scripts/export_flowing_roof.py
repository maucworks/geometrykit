# This file was generated with the assistance of an AI coding tool.
"""Export the flowing-roof patches (P-0 base + P-1 G1-constrained) to OBJ + IFC STEP.

Run in the oracle env (needs ifcopenshell + pythonocc for the constraint
normals); the OBJ export is occ-free.

    /opt/miniconda3/envs/a320-ifc-configurator/bin/python scripts/export_flowing_roof.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "flowing_roof", Path(__file__).resolve().parent.parent / "examples" / "flowing_roof.py"
)
fr = importlib.util.module_from_spec(_spec)
sys.modules["flowing_roof"] = fr
_spec.loader.exec_module(fr)

from geometrykit.export import surfaces_to_ifc_step, write_obj, write_stl  # noqa: E402
from geometrykit.runtimes._npu import patch_g1, patch_np  # noqa: E402


def main() -> None:
    params = {
        "height_sw": 2.5, "height_se": 2.5, "height_nw": 2.5, "height_ne": 2.5,
        "height_nnw": 2.5, "height_nne": 2.5, "twist": 25,
    }
    g = fr.build_curves(params)
    fr.build_ribbons(g, twist=params["twist"])
    fr.build_patches(g)

    out_dir = Path(__file__).resolve().parent.parent / "storage" / "flowing-roof-eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── P-1 G1-constrained patches (kernel) ──
    from geometrykit.geometry.surface import Surface as _Surface

    # sample the ribbon normals for the constraint fields (OCCT twin probe)
    curves = [g.bottom, g.right, g.top, g.left]
    names = ["south", "east", "north", "west"]
    fields = {}
    for name, curve in zip(names, curves):
        ribbon = g.ribbons.get({"south": "bot", "east": "right", "north": "top", "west": "left"}[name])
        if ribbon is None:
            continue
        field = []
        for k in range(11):
            t = (k + 1) / 12
            q = curve.point_at(t)
            best = None
            for ve in (0.0, 1.0):
                p = ribbon.point_at(t, ve)
                dist = abs(p - q)
                if best is None or dist < best[0]:
                    best = (dist, ve)
            dist, ve_s = best
            if dist < 1e-4:
                n_ = ribbon.normal_at(t, ve_s)
                ln = math.sqrt(n_.x**2 + n_.y**2 + n_.z**2) or 1.0
                field.append((n_.x / ln, n_.y / ln, n_.z / ln))
        fields[name] = field

    p1_surf, _ = patch_g1.constrained_coons_patch(curves, fields, nu=16, nv=16)
    print("P-1 patch built (G1-constrained, mixed heights)")

    # ── OBJ + STL export (occ-free): both G1 patches + all ribbons ──
    obj_path = out_dir / "flowing-roof-patches.obj"
    write_obj([p1_surf, g.surf2], str(obj_path), labels=["patch_g1", "patch_occ_twin"])
    print(f"OBJ → {obj_path}")

    stl_path = out_dir / "flowing-roof-patches.stl"
    write_stl([p1_surf, g.surf2], str(stl_path))
    print(f"STL → {stl_path}")

    # ── IFC STEP (oracle env) ──
    try:
        step_path = out_dir / "flowing-roof-patches.step"
        surfaces_to_ifc_step([p1_surf, g.surf2], str(step_path), labels=["PatchG1", "PatchOccTwin"])
        print(f"STEP → {step_path}")
    except ImportError:
        print("ifcopenshell not available — skipping STEP export")

    print("done — load the OBJ/STEP in a 3D viewer or CAD to evaluate")


if __name__ == "__main__":
    import math  # noqa: F401

    main()
