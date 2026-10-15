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
The engine split is **live and shipped**: `crates/geometrykit-core`
holds the kernels as pure Rust (`eval`, `basis`, `interp`, `linalg`,
`section`); `crates/geometrykit-py` is the maturin/marshalling layer;
`runtimes/dispatch.py` routes the kernel entries per engine selection.
Plain installs without a Rust toolchain stay mirror-only, by design.

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
- **Engine matrix:** every full-suite train runs in **both** engine
  configurations before a commit lands:
  `GEOMETRYKIT_ENGINE=rust pytest tests/ -q` and
  `GEOMETRYKIT_ENGINE=npu pytest tests/ -q` (plus the oracle run below).
  The dispatch routes through the extension when importable, so the
  two configurations genuinely exercise the two engines.
- **Oracle tier:** `conda env create -f environment.yml` then
  `conda run -n geometrykit-oracle python -m pytest tests/ -q`. That
  environment exists only for the tests that check the kernel against OCCT
  (`pythonocc-core` is conda-only); it is deliberately separate from the
  plain venv and from any other project's environment. Recreate it with
  `conda env update -f environment.yml --prune` when the pins move. The
  oracle env installs the same package from the same checkout, so it
  shares `python/geometrykit/_rust.abi3.so` when built.
- **Feature-level capability gates** live in the default suite
  (e.g. `tests/geometry/test_diagrid_capability.py`): a shipped
  workflow becomes a committed test — correctness invariants with
  *measured* bounds, engine-determinism across configurations, and a
  wall-clock budget. Bounds are recorded values, never vibes.
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

- The numpy mirror (`runtimes/_npu/`) is the API floor **and the
  reference implementation, forever**: every kernel function exists
  there before an accelerator port, and it is never removed.
- `crates/geometrykit-core` (pure Rust, `#![forbid(unsafe_code)]`) must
  never import pyo3; `crates/geometrykit-py` contains no geometry
  logic — only marshalling.
- **Port law:** a port is a *verbatim* translation of the mirror —
  same loop order, same expression association, same guards. Two
  languages must produce bit-identical floats on the same inputs;
  reassociation "to make it clean" is a bug.
- **Parity gate:** `tests/test_engine_parity.py` pins every ported
  kernel bit-for-bit against the mirror (hex-float bit patterns,
  MD5-16 contract signatures), including the shipped-config hail
  probes. A port lands only when its gate is green **and** the
  full-suite train above ran in both configurations.
- **Error contract is part of the mirror:** the wrapper maps core
  errors to the mirror's exact Python types and messages
  (`ZeroDivisionError` for zero weight sums, `RuntimeError` for
  singular interpolation systems, `ValueError` for bandwidth
  overflow). No Rust `panic!` may cross a Python boundary.
- **Routing:** `runtimes/dispatch.py` selects per engine
  (`geometrykit.engine.engine()`; `GEOMETRYKIT_ENGINE=npu` overrides
  to the mirror). Mirror call sites stay direct — only `dispatch` and
  the public graph routes flip.

## Building the Rust extension (dev)

- `pip install "maturin>=1,<2"`, then
  `python scripts/build_rust_extension.py` — builds the wheel via
  maturin and copies `python/geometrykit/_rust.abi3.so` into the
  package dir of the running interpreter (editable installs share the
  path across venvs). The `.so` is gitignored; never commit it.
- Plain `cargo test`/`cargo build` works for the workspace
  (`crates/.cargo` is at the repo root `.cargo/config.toml` — cargo
  reads config relative to the *working directory*, which matters
  because maturin invokes cargo with `--manifest-path`).
- The extension-wheel packaging layout (nested vs. top-level module)
  is a deferred Phase 3 decision; the build script is the supported
  dev route until then.
- Open (Phase 3): the dual-engine CI matrix and the accelerator-wheel
  distribution layout.

## Licensing
