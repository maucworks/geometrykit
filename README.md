# geometrykit

OCC-free NURBS geometry for the ifckit family: value types (`Vec`,
`Plane`, `Line`, `Arc`, `Polyline`, `Path`, `Curve`, `Surface`), NURBS
evaluation / tangents / normals / projection, energy-minimising patch
fills with G1/G2 continuity options, plane × surface sections with
pcurve tracks, and STEP/OBJ/STL writers. No OCCT, no compiled
toolchain required to install.

## Install

```bash
pip install geometrykit   # Python >= 3.9, incl. Rhino 8's CPython 3.9
```

Binary wheels (`cp39-abi3`) carry the compiled Rust core
(`geometrykit._rust`) where a platform build exists, and the public
API routes to it automatically. Platforms without a matching wheel get
the pure-Python/numpy mirror instead: same API, same results, slower.
`GEOMETRYKIT_ENGINE=npu` forces the mirror; the default prefers the
accelerator when importable.

## Quickstart

```python
from geometrykit import engine
from geometrykit.geometry import Curve, Surface
from geometrykit.geometry.primitives import Vec

curves = [
    Curve.from_tangents(start=Vec(0, 0, 2.4), tan_start=Vec(1, 0, 0),
                        end=Vec(8, 0, 2.8), tan_end=Vec(1, 0, 0), scale=3.0),
    Curve.from_tangents(start=Vec(8, 0, 2.8), tan_start=Vec(0, 1, 0),
                        end=Vec(8, 6, 2.9), tan_end=Vec(0, 1, 0), scale=3.0),
    Curve.from_tangents(start=Vec(0, 6, 3.2), tan_start=Vec(1, 0, 0),
                        end=Vec(8, 6, 2.9), tan_end=Vec(1, 0, 0), scale=3.0),
    Curve.from_tangents(start=Vec(0, 6, 3.2), tan_start=Vec(0, -1, 0),
                        end=Vec(0, 0, 2.4), tan_end=Vec(0, -1, 0), scale=3.0),
]
surface = Surface.patch(curves, fill="energy", nu=16, nv=16, tangency=0.75)
proj, (u, v) = surface.project(Vec(-1.0, -1.0, 3.0))
print(engine.engine(), proj, (u, v))
```

## What is in here

- NURBS evaluation, analytic tangents and normals, closest-point
  projection (kernel Newton, no OCC).
- Energy-minimising patch fill over a boundary ring
  (`Surface.patch(..., fill="energy")`), with tangency strength and
  optional G2 and adaptive re-fit.
- Plane × surface sections with parameter tracks, batch level sets,
  and node-exact diagrid weaving.
- Writers: STEP (AP214 B-rep), OBJ, STL.
- `geometrykit-core` on crates.io: the same kernels as a pure-Rust
  dependency (no bindings, no unsafe code).

Out of scope by design: B-rep booleans and tolerant topology (volumes
are OCCT's domain); IFC serialisation lives in ifckit.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
GEOMETRYKIT_ENGINE=npu .venv/bin/python -m pytest tests/ -q
GEOMETRYKIT_ENGINE=rust .venv/bin/python -m pytest tests/ -q
```

The oracle tier checks the kernel against OCCT and stays local
(`pythonocc-core` is conda-only; see AGENTS.md). Every suite train
runs in both engine configurations before a commit lands.

## Licensing

LGPL-3.0-or-later (`COPYING.LESSER`, `COPYING`). Kernel code promoted
from ifckit's geometry package carries a provenance header recording
its origin.
