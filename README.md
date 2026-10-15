# geometrykit

Geometry primitives and NURBS data model: `Vec`, `Plane`, `Curve`, `Surface`,
`Path`, and (planned) G1 constrained patches, curve/surface clipping, trim
pcurves, and a tessellator — the geometry substrate for the ifckit family.

## Status

Source of truth: `github.com/maucworks/geometrykit` — check it out anywhere;
wheels are built from a checkout root (`python -m build`, `packages.find`
over this layout). The package itself is unaware of check-out locations.

Releases so far (0.0.1/0.0.2) are a **name claim + placeholder**: the API is not yet
published, and the package intentionally exports nothing beyond
`geometrykit.__version__`. Version bumps arrive with the first real
API (planned v0.1: primitives promoted from `ifckit.geometry`, extracted
from the A320 configurator corpus with its existing test coverage as oracle
spec).

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/ -q                       # full suite (plain env)

# oracle tier: the tests that check the kernel against OCCT
conda env create -f environment.yml              # once
conda run -n geometrykit-oracle python -m pytest tests/ -q
# with coverage + the staged ladder:
conda run -n geometrykit-oracle python -m pytest tests/ --cov -q
conda run -n geometrykit-oracle python -m coverage json -o /tmp/geometrykit-cov.json
python scripts/coverage_ladder.py /tmp/geometrykit-cov.json
# corpus refresh (A320 demos → tests/fixtures/corpus/):
# where ifckit is importable (its own environment, not the oracle one):
python scripts/capture_corpus.py
```

Layers: `tests/geometry/` (ported ifckit geometry suite), `tests/test_contract.py`
(contract v0), `tests/test_engine_probe.py` + `tests/test_engine_parity.py`
(runtime split), `tests/oracle/` (OCCT goldens, skipif), `tests/panel/`
(panelization acceptance, M2 tripwire).

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

## Licensing

geometrykit is LGPL-3.0-or-later, aligned with ifckit. The kernel carries code
promoted from ifckit's geometry package; the provenance of every carried
module is documented in its file header.

## Sibling projects

- `ifckit` — elements/builders/model that bind `geometrykit` to IFC.
- The associativity engine (name TBD: `modelkit`/`stageskit`) — identity,
  stage signatures, dirty propagation; geometry-agnostic, stdlib-only.
