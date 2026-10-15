# This file was generated with the assistance of an AI coding tool.
"""AGENTS.md for geometrykit.

Conventions for AI agents and human contributors. ifckit-ecosystem rules
(frame-agnostic code, pinned dependencies, English everywhere) apply with
the specifics below.

## Project overview

geometrykit is the geometry substrate for the ifckit family: value types
(Vec/Plane/Line/Arc/Polyline/Path/Curve/Surface/…), the NURBS kernels
(evaluation, projection, intersection, tessellation), the G1 patch work,
and the engine split (`runtimes/_npu` mirror + optional `geometrykit._rust`
accelerator). It is IFC-free by design: IFC serialisation lives in ifckit,
which installs an IFC bridge (`geometrykit/geometry/_bridge.py`).

## Licensing

- **LGPL-3.0-or-later** (single license; `COPYING.LESSER`, `COPYING`).
- The kernel carries code promoted from ifckit's geometry package; every
  carried module has a provenance header recording its origin.

## Repo boundary (enforced)

- This package is **unaware of check-out locations**. Never mention, in
  code, docs, READMEs, commit messages, or anywhere else:
  check-out environments/admin layers, plan file paths or plan names
  (`plans/*`, `PLAN-*`, `STRATEGY-*`), the words "monorepo", "admin
  repo/environment/workspace/layer".
- Enforcement: `python tools/check_boundary.py` — scans tracked content
  and pending commit messages; also runs as
  `tests/test_repo_boundary.py` inside the default pytest suite.
- Incoming plan/strategy documents live outside this repo. Never import
  them here, never reference them here; discuss cross-repo work in the
  PR description only.

## Language

All written content — code, docstrings, comments, docs, commit messages —
is in **English**.

## Commits

- Same convention as ifckit: subject ≤50 chars, imperative mood, body
  notes AI-generated changes ("Generated with the assistance of an AI
  coding tool."), new files carry the generated-code header comment.
- Never push to `master/master` directly: feature branches + PRs.

## Code style

- ruff, line length 100, target py39 (the promoted legacy modules are
  excluded from the docstring rule set for now; new code follows
  Google-style docstrings).
- Run `ruff check` before committing; tests before pushing.

## Testing

- `python3 -m venv .venv && pip install -e ".[dev]"; pytest tests/ -q`
- **Oracle tier:** `conda env create -f environment.yml` then
  `conda run -n geometrykit-oracle python -m pytest tests/ -q`. That
  environment exists only for the tests that check the kernel against OCCT
  (`pythonocc-core` is conda-only); it is deliberately separate from the
  plain venv and from any other project's environment. Recreate it with
  `conda env update -f environment.yml --prune` when the pins move.
- **Generated results and scratch work go in the gitignored `wip/`
  directory** -- exported STEP/OBJ/STL, viewer pages, sample outputs, anything
  meant to be looked at rather than committed. Library tests stay in
  `tests/`; a script may write to `wip/`, never to a tracked path.
- Oracle-inclusive coverage run + the staged coverage ladder
  (`scripts/coverage_ladder.py`) documented in README § Development.
- Fixtures: the A320 demo corpus is committed under `tests/fixtures/`;
  refresh with `scripts/capture_corpus.py` where ifckit and deps exist.
- Dependency pins follow the ifckit principle: `>= min-proven, <= current`,
  never open-ended.

## Engine rules

- The numpy mirror (`runtimes/_npu/`) is the API floor — every kernel
  function exists there before an accelerator port.
- `crates/geometrykit-core` (pure Rust) must never import pyo3;
  `crates/geometrykit-py` contains no geometry logic — only marshalling.
- Cross-engine parity and contract-hash equality are CI property tests;
  divergence from the mirror is a test failure, never a feature.
