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
