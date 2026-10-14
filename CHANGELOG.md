# v0.0.1 — 2026-10-13

- Name claim on PyPI + placeholder distribution. No API yet; the public
  contract is documented in README.md (roadmap) and ships with v0.1.

# v0.2.0 — 2026-10-14

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
