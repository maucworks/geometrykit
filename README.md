# geometrykit

Geometry primitives and NURBS data model: `Vec`, `Plane`, `Curve`, `Surface`,
`Path`, and (planned) G1 constrained patches, curve/surface clipping, trim
pcurves, and a tessellator — the geometry substrate for the ifckit family.

## Status

This release (0.0.1) is a **name claim + placeholder**: the API is not yet
published, and the package intentionally exports nothing beyond
`geometrykit.__version__`. Version bumps arrive with the first real
API (planned v0.1: primitives promoted from `ifckit.geometry`, extracted
from the A320 configurator corpus with its existing test coverage as oracle
spec).

## Roadmap (contract-first, per the kernel plan)

- v0.1 — value-type primitives (`Vec`, `Plane`, `Arc`, `Line`, `Path`,
  `Curve`, `Surface`); `ifckit.geometry` becomes a re-export facade.
  Data contract: id + plain arrays (`{kind, degrees, knots, cps}(m,4)`,
  homogeneous `[w-mu, w-nu, w-rho, w]`).
- v0.2+ — G1 constrained filling (Gregory corners) over polygonal charts;
  Bezier-column clipping (plane-surface, closest-point); trims (pcurves);
  tessellator (curvature-driven).
- Out of scope by design: B-rep booleans, tolerant topology (the kernel is
  B-rep-free on purpose; volumes are ifcopenshell/OCCT's domain).

## Sibling projects

- `ifckit` — elements/builders/model that bind `geometrykit` to IFC.
- The associativity engine (name TBD: `modelkit`/`stageskit`) — identity,
  stage signatures, dirty propagation; geometry-agnostic, stdlib-only.
