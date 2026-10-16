# v0.6.0 — 2026-10-16: tessellation, arc-length division, panel
# metrics (Python), plus their Rust-core ports (core 0.2.0).
# The wheel matrix carries the accelerator per platform (abi3-py39,
# incl. Rhino 8's CPython 3.9); the pure wheel stays the fallback.

# v0.5.1 — 2026-10-16: the rewritten README reaches PyPI (the 0.4.0/
# 0.5.0 pages still described the 0.0.x name-claim placeholder). No
# code changes.

- **Panel metrics promoted** (`geometry.panel`: `FieldReport`,
  `cells_report`, `field_report`): the Eigensatz buildability metrics
  move here from ifckit (pure stdlib, no twin); exported from
  `geometrykit.geometry` with tests. The old package comment claiming
  panel_report stays in ifckit is revised with it.

- **Kernel arc-length division** (`curve_np.arclength_divide`,
  `TestDivideArcLength`): the OCC-free fallback of `Curve.divide`
  sampled uniformly in parameter space (measured 2.7x gap spread);
  the chord-table inversion holds arc-length uniformity within a
  few percent. The mirror function is directly testable and the
  port target; `divide()` itself keeps taking the OCC path when
  available (pinned with path-agnostic bounds).
- **Phase 2 port 8: arc-length division in core** (`curve::Curve`
  point eval + `arclength_divide`, `curve_divide_grid` batch
  marshalling, `dispatch.arclength_divide`): bit-parity proven
  mirror vs. extension on wiggle/line/kinked curves (one 1-ulp
  association fix along the way: `(lo-1+frac)/n`, not split).
  Suites: plain 714/0, oracle 714/0.

- **Adaptive kernel tessellation** (`runtimes/_npu/tessellate_np`,
  `test_tessellate.py`): OCC-free surface meshing — a restricted
  quadtree over the parameter domain (dyadic lattice stations, one
  basis build per eval), error-split on edge-midpoint deviation vs.
  `deflection` and corner-normal deviation vs. `angle_tol`, 2:1
  balance, green closure fanning split edges from the evaluated
  centroid. Every vertex is an evaluated surface point, every edge
  shared exactly: watertight by construction, deflection measured
  exactly through the returned uv tracks (no projection Newton in
  the loop). Returns `(verts, normals, tris, uvs)`, house-style like
  section chains. Measured: saddle 32 tris at 0.01 (worst 5.6e-03),
  roof fill 140 verts / 200 tris in 0.02 s (worst 7.6e-03).
  Suites: plain 700/0.
- **Phase 2 port 7 + dispatch cutover**: the tessellator in
  `geometrykit-core::tessellate` (`tessellate_grid` batch
  marshalling, `dispatch.tessellate`), bit-parity proven mirror vs.
  extension on the saddle and its rational twin. `to_mesh_dict` and
  `preview` gain `method="adaptive"` next to the unchanged default
  `method="occ"` (same dict shape; unknown methods raise).
  Suites: plain 706/0, oracle 706/0.

# v0.5.0 — 2026-10-16: the precompiled wheel matrix (pip installs the
# rust engine per platform; abi3-py39 covers Rhino 8's CPython 3.9.11)
# plus the corrected binary-wheel metadata (Root-Is-Purelib=false, the
# RECORD self-row). Pure wheel unchanged as the universal fallback.

# v0.4.0 — 2026-10-15 (first upload under this version; the 0.2.0/0.3.0
# milestones below were internal trains without a PyPI upload)

- **Rust scaffold, Phase 1** (`crates/`): the engine split now exists.
  `crates/geometrykit-core` holds the kernels in pure Rust (no pyo3,
  `#![forbid(unsafe_code)]`), starting with the contract float layer
  (`bits_hex`, bit-identical to `contract.float_to_hex` incl. signed
  zeros and NaN payloads); `crates/geometrykit-py` is pyo3 marshalling
  only and builds via maturin as `geometrykit._rust`
  (`scripts/build_rust_extension.py`). The engine probe prefers the
  extension when importable, `GEOMETRYKIT_ENGINE=npu` forces the
  mirror. Parity harness (`test_engine_parity.py`) proves signature
  stability across two builds of the same kernel data and, with the
  extension present, cross-language float-bit agreement on every
  kernel result — the gate Phase 2 ports land against.
- **Phase 2 port 1: evaluation kernels** (`geometrykit-core::eval`,
  `geometrykit._rust.eval_point_grid/eval_all_grid`): the Piegl basis
  build, point, and both analytic tangents ported verbatim from the
  mirror (same A2.1/A2.2 spans, right-limit derivative convention, same
  accumulation order). Mirror vs. extension compared bit-for-bit on a
  deterministic saddle patch and its rational twin, including clamped
  stations and both domain ends: every IEEE pattern equal. Batch
  marshalling: one Python round-trip per grid, not per station.
- **Phase 2 port 2: projection** (`project_point`, `_newton`,
  `_edge_min` core port + `geometrykit._rust.project_grid`): the P0.1
  algorithm verbatim -- coarse point-only grid, distinct-seed damped
  Gauss-Newton (early-gradient break, degenerate-det break, five-step
  halving, domain clamp), then the four boundary isocurves by
  golden-section, global best kept. Mirror vs. extension bit-for-bit
  on structured targets (corner minima, edge grazing, interior curl)
  AND an 80-target fixed-seed shower across [-1.5, 2.5]^3: every
  (point, u, v) float identical.
- **Phase 3 cutover (dispatch)** (`runtimes/dispatch.py`,
  `Surface.point_at/tangents_at/normal_at/project`): the public kernel
  API routes through `engine.engine()` -- the `_rust` extension when
  importable, the mirror otherwise. Routed-vs-mirror bit-identity is a
  dedicated parity class (`TestRoutedParity`) plus suite runs in both
  engine configurations. Error contract preserved (zero weight-sum
  `ZeroDivisionError` from both engines; core returns the mirror's
  message, the wrapper maps it -- no Rust panic crosses Python).
  Timing on the shipped front patch (nu=16, G2+fill+adaptive, oracle
  env): 0.717 s routed vs 0.754 s mirror -- the residual per-call
  saving P0's cache did not already collapse; row assembly remains the
  dominant share and is port 3's target.
- **Phase 2 port 3: batched basis rows** (`geometrykit-core::basis`,
  `geometrykit._rust.basis_row_grid`, `dispatch.basis_row_list`): the
  row-assembly primitive of the G1/G2 edge systems batched over
  stations (verbatim `_nonzero_basis` port: same A2.2, same edge-aware
  span, same 1e-12 guards, mirror dict order preserved for `.items()`
  consumers). All five `patch_g1` call sites cut to the batched
  dispatch. Parity: single- and multi-span knot vectors, degrees 2-3,
  30 stations including duplicates, interior knots and domain ends --
  every id and float bit-identical to `curve_np._nonzero_basis`. Suite
  configs: plain 677/0 x2, oracle 677/0. Timing, shipped front patch:
  0.682 s routed vs 0.750 s mirror (the cutover gain doubled with the
  batched rows).
- **Phase 2 port 4: banded solvers** (`geometrykit-core::linalg`,
  `geometrykit._rust.banded_cholesky_grid/banded_solve_grid`,
  `dispatch.banded_cholesky/banded_solve`): `_as_band`/`banded_cholesky`
  /`banded_solve` verbatim (same upper-band storage, same 1e-14 pivot
  tolerance, mirror-form error messages; no production call site today
  -- engine completeness for the contract). Parity: 24-point
  tridiagonal factor + solve bit-for-bit, non-SPD and beyond-band
  errors raised identical from both engines. Tests corrected along the
  way: the mirrored lower entry belongs in row i+1 (its skipping is
  the property being tested), and source rows stay pair-lists, not
  dicts, through the extension. Suites: plain 686/0 x2 configs; core
  13/13.
- **Diagrid capability suite** (`tests/geometry/test_diagrid_capability.py`):
  the shipped pcurve-diagrid workflow as a committed gate on the
  flowing-roof network -- energy fill, two diagonal plane families at
  50 batch levels, weave (48+48 curves, 1128 nodes), unit-length
  normal spikes at every node, node-plane residuals vs. both families
  (worst 3.4e-05, measured), two-level refined node-on-curve probe
  (worst 1.2e-03, measured) with the sampling-resolution trap
  documented, engine-config determinism of the node set, and a
  wall-clock budget (< 8 s; shipped measurement ~1.5-2 s). Plain
  kernels only: no OCC, no numpy.
- **Weave crossing index** (`section_np.weave_diagrid`): the B family's
  segments get a 2D bucket grid (cell 0.05, wide segments fall back to
  pairwise probing), so each A segment only tests the B segments whose
  boxes share a cell. Measured on the shipped diagrid: the 2D crossing
  search 845 -> 83 ms, weave total 1068 -> 302 ms; full workflow
  (export incl. sections, weave, 1129 nodes, 1128 normal spikes, two
  STEP writes) 1.95 -> 1.12 s. One near-duplicate twin node now
  resolves to the other member of its pair (1128 -> 1129 nodes); all
  invariant gates hold (plane residual, exactness, determinism).
- **Phase 2 ports 5+6: section marching + curve interpolation in core**
  (`geometrykit-core::section`, `::interp`,
  `geometrykit._rust.section_levels_grid/interpolate_grid`,
  `dispatch.plane_section_chains/interpolate`): the level-set
  marching (field grid, saddle-cell pairing, greedy chaining, Newton
  polish) and the A9.1/A9.2 global interpolation (chord params,
  averaged knots, full-Jordan dense solve) ported verbatim.
  `Surface.section_levels` and the weave's knot assembly ride the
  extension when present. One chaining bug was caught by the parity
  gate itself: the mirror branches prepend/append on the loop token,
  not position -- they disagree when a degenerate first segment
  collapses the chain to one point (measured 57 vs 58 points on the
  A-family level 14). Engine determinism restored and suite-tested.
  Measured on the shipped diagrid: sections 551 -> 135 ms, weave 302
  -> 91 ms, whole exported workflow 1.95 -> 0.44 s. Suites: plain
  691/0 x2 configs; core 18/18.

- **Energy-minimising plate fill** (`patch_np.discrete_plate_fill`,
  `linalg_np.plate_fill`): the interior is the thin-plate solution through
  the boundary ring, not the Coons blend. Simply-supported by default;
  clamped where slope data exists (G1 as a boundary condition: fill alone
  leaves every edge at ~95% of the support rate); ring-2 curvature belts
  opt-in behind `g2_in_fill` (measured a regression for flat supports, kept
  as exact machinery). Analytic datums (`x*y`, `x**2`) reproduced to 1e-13.
- **Adaptive constrained approximation** (`fit_np.adaptive_fit`,
  `constrained_patch(..., adaptive=<tol>)`): rows-then-fit onto a
  C1-segmented degree-4 net with station boundaries held exact by
  equality elimination (1.6e-14). The user's eye judged it best of all
  variants; the entry path reproduces the file byte-identically.
- **Corner protocol** (`patch_g1.corner_report` with the cage datum,
  `corner_stations=0` opt-in): adjacent-station residual 2.8e-02 -> 1.4e-12
  at shipped density; zebra passes. Library default stays 1 (coarse
  density needs it); the driver opts in explicitly.
- **Public routing** (`Surface.patch(fill="energy")`): the whole continuity
  pipeline OCC-free (pure ribbons with the spine at 0.00e+00, kernel field
  reading, fill, rows) -- G1 1e-13 through the public route in the plain
  venv. Legacy default (`fill=None`, the OCCT twin) unchanged.
- **Measured table, flowing roof, driver density (nu=16):** cross-direction
  curvature peak 5.96 (Coons/chord) -> 1.25 (M2) -> 0.165 (M3) with twin at
  0.018; deviation from twin 0.175 -> 0.049 (M3) -> 0.022 (shipped,
  tangency 0.75 + adaptive); boundary exactness 1e-14 throughout; G1
  1e-13, curvature gaps <= 4e-3.

# v0.3.0 — 2026-10-14

- **Patch kernel P-1** (G1-constrained filling, `runtimes/_npu/patch_g1.py`):
  the tangent-plane constraint reduces to a linear condition on the control
  net — at a boundary station the patch's cross-derivative must satisfy
  `d . N = 0` with `N` the constraint surface's (ribbon's) unit normal.
  All four edges are assembled into **one** ridge-regularised least-squares
  system (`(A^T A + lambda I) D = A^T b`, ridge 1e-10 relative, one step of
  iterative refinement), so the edges no longer clobber each other around the
  corners and an already-satisfying field leaves the Coons base untouched
  (the coplanar case is bit-stable). Fields may be callables in patch
  parameter space, `(param, normal)` pairs, or plain normals aligned with the
  interior stations — equivalent forms give identical patches.
  Corner band (`corner_stations`, default 1 per end) is the documented zone
  where the boundary curves pin the cross-derivative; corner twist
  reconciliation is the next increment.
  Measured on the mixed-height flowing-roof fixture against OCCT ribbon
  normals: G1 residual 3e-9..7e-8 on cross-derivatives of magnitude ~9
  (tangent-plane agreement to ~1e-8 rad); max interior deviation from the
  OCCT `MakeFilling` twin 0.30 on an 8x6 patch.
- **Patch kernel P-0 follow-up** (`patch_np`): the station parameters of a
  Coons patch are now first-class (`coons_parametrization`), with
  `edge_curve_params` mapping patch stations onto an edge curve's own
  parameter (drawn direction may run opposite to the sampled one). G1
  constraints and their verification probes the same parameters the patch
  was built on.
- **Symmetric support ribbons** (`Surface.ribbon(..., symmetric=True)`,
  `patch_g1.support_fields`): the spine runs down the middle of the ribbon,
  so one support serves a patch on either side of the curve -- which is what
  a shared ridge needs, and it makes the constraint independent of which way
  the support was drawn. The rails are offsets of the spine's own control
  net, so the spine lies *exactly* on the middle line. `support_fields`
  returns the support's normal and its own slope along the spine, located by
  projection, so one-sided, symmetric and full supports all work without
  guessing an edge. `Surface.patch` now hands OCCT's filler the support
  face's own spine edge (a rebuilt edge that merely coincides with the
  boundary segfaults `MakeFilling`), bounding the face at the spine when the
  spine is interior.
- **Tangency strength** (`patch_g1`, `tangency=` on
  `solve_g1_constraints` / `constrained_coons_patch`): G1 fixes the tangent
  plane and leaves the boundary cross-derivative's *magnitude* to the solver,
  which typically under-delivers against a CAD tangency support. The optional
  factor names that magnitude as a multiple of the support's own slope,
  measured per unit distance so it is parameterisation-independent: `1.0`
  reproduces the support's own rate of departure, higher is stiffer, `None`
  (default) keeps the previous minimum-norm behaviour. Linear in the factor
  and G1-exact at every value (measured worst |d . N| 3.4e-08).
- **G2 (curvature) continuity** (`patch_g1`, `g2=True` on
  `constrained_coons_patch`): G1 matches the tangent plane and leaves the
  surface free to bend away from it immediately, which is visible as a crease
  against a flat support. G2 additionally matches the normal-section curvature
  across the boundary -- `(S_vv . N) = kappa_support * |S_v|^2`, linear in the
  poles once `|S_v|` and the support's curvature are taken from the G1 result,
  so it runs as a second pass over a wider control band with the G1 rows
  rebuilt alongside. Measured on the flowing roof against flat ribbons
  (support curvature exactly zero): the ridge's cross-curvature gap falls from
  0.184 to 1e-5, the other edges from ~0.02 to 1e-5, and G1 tightens from
  3e-08 to 3e-14 because both sets of rows are solved together. Needs the
  support and spine in the field spec (`patch_g1.support_constraint`), and
  composes with the tangency strength. The second-derivative weights
  (`_second_deriv_weights`) are validated against a position-based second
  difference.
- **Patch export** (`geometrykit/export.py`, `scripts/export_flowing_roof.py`):
  occ-free OBJ (quads) and STL writers, plus an **AP214 STEP** writer that
  emits real `B_SPLINE_SURFACE_WITH_KNOTS` faces -- control net, distinct
  knots and multiplicities, edges as the surface's own boundary curves with
  their pcurves -- inside the `SHELL_BASED_SURFACE_MODEL` /
  `MANIFOLD_SURFACE_SHAPE_REPRESENTATION` chain. That is the exchange a CAD
  kernel opens (Rhino, CATIA, SolidWorks); IFC-in-STEP is not. Rational
  surfaces are refused rather than silently written unweighted. Verified by
  reading the file back with OCCT: identical pole coordinates (1e-12) and
  identical evaluated points (1e-9) on the flowing-roof patch, and both
  patches of the roof arrive in one file. `write_step_geometry` also exports
  standalone curves (`GEOMETRIC_CURVE_SET` of `B_SPLINE_CURVE_WITH_KNOTS`), so
  a patch travels with the edge curves it was built from and every entity is
  named. The driver writes one STEP per run: the kernel patch, its four
  constraint ribbons and the seven network edge curves, in the mixed-height
  stress case, with the G1 residuals and the twin deviation reported on
  stdout (the OCCT twin is measured, not exported).

# v0.0.1 — 2026-10-13

- Name claim on PyPI + placeholder distribution. No API yet; the public
  contract is documented in README.md (roadmap) and ships with v0.1.

# v0.2.0 — 2026-10-14

- **Patch kernel P-0** (plan: OCC-exit §5): `Surface.patch` with position-only
  constraints routes to the occ-free discrete Coons kernel (`runtimes/_npu/patch_np.py`): exact corner pass-through, station-exact
  boundaries, coplanar-reflexive G1. Continuity-constrained patches
  (G1 + supports) keep the OCCT route until P-1 (sampled cross-field
  constraint rows) lands. Flowing-roof standalone example ported into
  `examples/` as the stress case (7 curves, 6 corners, stages documented).
- Spline-cage data model (contract v0.1): `CoordinateSystem` (placed
  Cartesian frame: position + Hamilton quaternion, canonical sign `qw ≥ 0`,
  XY plane = planarity plane for cage tangents) and `Corner` (associates a
  frame with curve terminals by angle records; `tangent_for` returns the
  3D world-space tangent; terminal-spacing degeneracy raises).
  `geometrykit.cage` module; serialization via the contract (hex floats).
- **M2 + M4-early (plan: OCC-exit)**: occ-free evaluation core — tensor
  Cox–de Boor `Surface.eval`/`tangents_at`/`normal_at` (homogeneous,
  rational-safe, correct one-sided edge derivatives) and kernel
  `Surface.project` (multi-start Gauss–Newton). field/patterns lose their
  `require_occ` gates (trim/crossing grids + Hankin cells run occ-free);
  `_intersect_uv` is a pure 2D solve. OCCT-verified on the corpus (eval
  to 1e-9, normals, projection minima) — OCCT remains oracle-only.
  Panel acceptance tests promoted to strict pass.

# v0.1.0 — 2026-10-13

- First API release (feature/contract): contract v0 (canonical form,
  hex-float signatures, element records), engine probe (`_npu` floor +
  future `_rust` accelerator), Tier-1 geometry promotion from ifckit
  (primitives/transform/path/curve/surface/frames/biarc/spiral/subdivision/
  field/patterns/intersection; IFC serialisation stays behind an optional
  host bridge), and the four-layer test suite (legacy geometry suite,
  contract, probe/parity, oracle, panel acceptance) with the A320 corpus
  (827 element records) and the staged coverage ladder.
- Relicensed from MIT to **LGPL-3.0-or-later** (`COPYING.LESSER` +
  `COPYING`), aligning with ifckit — the kernel carries code promoted
  from ifckit's geometry package.

# v0.0.2 — 2026-10-13

- Repository URLs filled in (github.com/maucworks/geometrykit); metadata
  re-release. No API change.
