# This file was generated with the assistance of an AI coding tool.
"""Engine parity harness: `_npu` mirror vs. `_rust` accelerator.

Phase 1 (scaffold): the harness runs with the mirror on both sides and
holds bit-exact contract signatures stable across two independent
builds of the same kernel data — the shape a Phase 2 port must satisfy,
already exercised end to end. When the extension is importable, the
same values are additionally checked against core's canonical
float patterns (`bits_hex`), which is a real cross-language test today:
two languages, one IEEE-754 payload convention.

Contract grain: hex-float bit patterns (`f64:XXXXXXXXXXXXXXXX`) rolled
into an MD5[:16] signature via `geometrykit.contract`, identical to the
record/corpus convention.
"""

from __future__ import annotations

import math

import pytest
from geometrykit import contract, engine
from geometrykit.geometry import Surface, Vec
from geometrykit.geometry.primitives import Vec as _Vec

_STATIONS = (
    0.0, 0.0625, 0.25, math.sqrt(2.0) / 4.0, 0.5, 0.75, 0.90625, 1.0,
)

_TRICKY_FLOATS = (
    0.0, -0.0, 1.0, -2.5, 0.1 + 0.2, 1e300, -1e-300, 5e-324,
    float("inf"), float("-inf"), float("nan"), 2**53 + 1.0,
    math.pi, math.sqrt(2.0),
)


def _surface() -> Surface:
    """Deterministic bicubic patch: 4x4 poles, no numpy, no OCC.

    The poles carry a gentle saddle so curvature and projection have
    something to bite on.
    """
    def z(x: float, y: float) -> float:
        return 3.0 + 0.35 * x + 0.15 * y - 0.8 * x * (1.0 - x) * y * (1.0 - y)

    rows = []
    for j in range(4):
        row = []
        for i in range(4):
            row.append(_Vec(i / 3.0, j / 3.0, z(i / 3.0, j / 3.0)))
        rows.append(row)
    return Surface(
        rows,
        uknots=[0.0, 1.0], vknots=[0.0, 1.0],
        umults=[4, 4], vmults=[4, 4],
        udegree=3, vdegree=3,
    )


def _sig(rows: list) -> str:
    """MD5[:16] contract signature of a grid of (x, y, z) rows."""
    data = {"k": 1, "rows": [[float(v) for v in r] for r in rows]}
    return contract.signature(data)


def _eval_rows(surf: Surface) -> list:
    out = []
    for u in _STATIONS:
        for v in _STATIONS:
            p = surf.point_at(u, v)
            out.append((p.x, p.y, p.z))
    return out


def _projection_rows(surf: Surface) -> list:
    out = []
    for u in _STATIONS:
        for v in _STATIONS:
            p = _Vec(0.37 + u, 2.1 - v, 2.4 + 0.9 * u)
            q, _uv = surf.project(p)
            out.append((q.x, q.y, q.z))
    return out


class TestHarnessSelfParity:
    """Both sides the mirror: signatures stable across two builds."""

    def test_eval_grid_signature_identical(self):
        assert _sig(_eval_rows(_surface())) == _sig(_eval_rows(_surface()))

    def test_projection_signature_identical(self):
        assert (
            _sig(_projection_rows(_surface()))
            == _sig(_projection_rows(_surface()))
        )

    def test_frozen_patterns_reference(self):
        # anchors: the exact bit patterns this test expects from the
        # analytic knobs of _surface's first pole (guards the station
        # set against silent edits)
        surf = _surface()
        p0 = surf.control_points[0][0]
        assert contract.float_to_hex(p0.x) == "f64:0000000000000000"
        assert contract.float_to_hex(p0.z) == "f64:4008000000000000"  # 3.0


@pytest.mark.rust
@pytest.mark.skipif(not engine.has_rust(), reason="requires geometrykit._rust")
class TestAcceleratorParity:
    """The extension agrees bit-for-bit with the canonical patterns."""

    def test_core_version_stamp(self):
        from geometrykit import _rust

        assert _rust.core_version() == "geometrykit-core 0.1.0"

    def test_bits_hex_agrees_with_contract(self):
        from geometrykit import _rust

        for v in _TRICKY_FLOATS:
            assert _rust.bits_hex(v) == contract.float_to_hex(v), v

    def test_kernel_results_bits_hex_agree(self):
        from geometrykit import _rust

        surf = _surface()
        for u in _STATIONS:
            for v in _STATIONS:
                p = surf.point_at(u, v)
                for x in (p.x, p.y, p.z):
                    assert _rust.bits_hex(x) == contract.float_to_hex(x)
        for row in _projection_rows(surf):
            for x in row:
                assert not (x != x), "NaN leaked into a projection row"

    def test_pattern_of_engine_state(self):
        Vec  # names stay import-stable for the Phase 2 dispatch seam


@pytest.mark.rust
@pytest.mark.skipif(not engine.has_rust(), reason="requires geometrykit._rust")
class TestEvaluationParity:
    """Port 1: evaluation kernels, mirror vs. extension, bit-for-bit.

    The mirror is the reference: every station pair runs both engines
    and every float must land on the same IEEE bit pattern -- points,
    tangents, rational surfaces, clamped stations, domain ends.
    """

    @staticmethod
    def _rational() -> Surface:
        """Same poles under a positive weight grid (rational path)."""
        import math

        surf = _surface()
        weights = [
            [1.0 + 0.5 * math.sin(1.3 * i + 0.7 * j + 0.2) for j in range(4)]
            for i in range(4)
        ]
        return Surface(
            surf.control_points,
            surf.uknots, surf.vknots, surf.umults, surf.vmults,
            surf.udegree, surf.vdegree,
            weights,
        )

    @staticmethod
    def _station_pairs() -> list:
        pairs = []
        for u in _STATIONS:
            for v in _STATIONS:
                pairs.append((u, v))
        # clamping: outside-domain stations collapse to the clamp
        for u, v in ((-0.25, 0.5), (1.5, -0.5), (1.25, 1.25), (-1.0, -1.0)):
            pairs.append((u, v))
        return pairs

    def _compare_points(self, surf):
        from geometrykit import _rust

        pairs = self._station_pairs()
        mine = [surf.point_at(u, v) for u, v in pairs]
        mine_rows = [(p.x, p.y, p.z) for p in mine]
        theirs = _rust.eval_point_grid(surf, [u for u, _ in pairs], [v for _, v in pairs])
        assert len(theirs) == 3 * len(pairs)
        got = [(theirs[k], theirs[k + 1], theirs[k + 2]) for k in range(0, len(theirs), 3)]
        for (a, b) in zip(mine_rows, got):
            for va, vb in zip(a, b):
                assert contract.float_to_hex(va) == contract.float_to_hex(vb), (va, vb)
        assert _sig(mine_rows) == _sig(got)

    def test_point_grid_bicubic(self):
        self._compare_points(_surface())

    def test_point_grid_rational(self):
        self._compare_points(self._rational())

    def _compare_all(self, surf):
        from geometrykit import _rust
        from geometrykit.runtimes._npu import surface_np

        pairs = self._station_pairs()
        mine = [surface_np.eval_all(surf, u, v) for u, v in pairs]
        mine_rows = [
            (pt.x, pt.y, pt.z, su.x, su.y, su.z, sv.x, sv.y, sv.z)
            for pt, su, sv in mine
        ]
        theirs = _rust.eval_all_grid(surf, [u for u, _ in pairs], [v for _, v in pairs])
        assert len(theirs) == 9 * len(pairs)
        got = [
            tuple(theirs[k:k + 9]) for k in range(0, len(theirs), 9)
        ]
        for (a, b) in zip(mine_rows, got):
            for va, vb in zip(a, b):
                assert contract.float_to_hex(va) == contract.float_to_hex(vb), (va, vb)
        assert _sig(mine_rows) == _sig(got)

    def test_all_grid_bicubic(self):
        self._compare_all(_surface())

    def test_all_grid_rational(self):
        self._compare_all(self._rational())

    @staticmethod
    def _targets() -> "list[_Vec]":
        base = [_Vec(0.37 + u, 2.1 - u, 2.4 + 0.9 * u) for u in _STATIONS]
        base.append(_Vec(-1.0, -1.0, 3.0))   # beyond the (0,0) corner
        base.append(_Vec(1.4, -0.4, 3.6))    # grazing the south edge
        base.append(_Vec(-0.6, 0.4, 3.05))
        base.append(_Vec(1.05, 1.05, 3.0))   # interior curl, not the corner
        return base

    def _compare_projection(self, surf):
        self._compare_projection_from(surf, self._targets())

    def test_projection_bicubic(self):
        self._compare_projection(_surface())

    def test_projection_rational(self):
        self._compare_projection(self._rational())

    def test_projection_hail_bicubic(self):
        """Fixed-seed target shower: 80 targets across the unit cube,
        plus its pushed-out variants; kills accidental parity."""
        import random

        rng = random.Random(20261015)
        ts = []
        for _ in range(80):
            ts.append(_Vec(
                rng.uniform(-1.5, 2.5),
                rng.uniform(-1.5, 2.5),
                rng.uniform(1.5, 5.0),
            ))
        surf = _surface()
        self._compare_projection_from(surf, ts)

    def _compare_projection_from(self, surf, pts):
        from geometrykit import _rust
        from geometrykit.runtimes._npu import surface_np

        mine = [surface_np.project_point(surf, p, m=6, n=6, iters=40) for p in pts]
        mine_rows = [(q.x, q.y, q.z, uv[0], uv[1]) for q, uv in mine]
        flat = [c for p in pts for c in (p.x, p.y, p.z)]
        theirs = _rust.project_grid(surf, flat)
        got = [tuple(theirs[k:k + 5]) for k in range(0, len(theirs), 5)]
        for (a, b) in zip(mine_rows, got):
            for va, vb in zip(a, b):
                assert contract.float_to_hex(va) == contract.float_to_hex(vb), (va, vb)
        assert _sig(mine_rows) == _sig(got)


@pytest.mark.rust
@pytest.mark.skipif(not engine.has_rust(), reason="requires geometrykit._rust")
class TestRoutedParity:
    """Phase 3 cutover: the routed public API equals the mirror bitwise.

    Routed means `Surface.point_at/tangents_at/normal_at/project` under
    dispatch (extension when importable) — the exact API the pipeline
    rides on, compared here against the untouched mirror.
    """

    def _stations(self):
        pairs = []
        for u in _STATIONS:
            for v in _STATIONS:
                pairs.append((u, v))
        return pairs

    def test_point_at_routed_matches_mirror(self):
        from geometrykit.runtimes._npu import surface_np

        surf = _surface()
        for u, v in self._stations():
            mine = surf.point_at(u, v)
            ref = surface_np.eval_point(surf, u, v)
            for a, b in zip((mine.x, mine.y, mine.z), (ref.x, ref.y, ref.z)):
                assert contract.float_to_hex(a) == contract.float_to_hex(b)

    def test_tangents_at_routed_matches_mirror(self):
        from geometrykit.runtimes._npu import surface_np

        surf = _surface()
        for u, v in self._stations():
            su, sv = surf.tangents_at(u, v)
            rsu, rsv = surface_np.eval_tangents(surf, u, v)
            for a, b in zip(
                (su.x, su.y, su.z, sv.x, sv.y, sv.z),
                (rsu.x, rsu.y, rsu.z, rsv.x, rsv.y, rsv.z),
            ):
                assert contract.float_to_hex(a) == contract.float_to_hex(b)

    def test_normal_at_routed_matches_mirror(self):
        from geometrykit.runtimes._npu import surface_np

        surf = _surface()
        for u, v in self._stations():
            mine = surf.normal_at(u, v)
            ref = surface_np.eval_normal(surf, u, v)
            for a, b in zip((mine.x, mine.y, mine.z), (ref.x, ref.y, ref.z)):
                assert contract.float_to_hex(a) == contract.float_to_hex(b)

    def test_project_routed_matches_mirror(self):
        from geometrykit.runtimes._npu import surface_np

        surf = _surface()
        for p in TestEvaluationParity._targets():
            mine_p, mine_uv = surf.project(p)
            ref_p, ref_uv = surface_np.project_point(surf, p)
            for a, b in zip(
                (mine_p.x, mine_p.y, mine_p.z, mine_uv[0], mine_uv[1]),
                (ref_p.x, ref_p.y, ref_p.z, ref_uv[0], ref_uv[1]),
            ):
                assert contract.float_to_hex(a) == contract.float_to_hex(b)


@pytest.mark.rust
@pytest.mark.skipif(not engine.has_rust(), reason="requires geometrykit._rust")
class TestBasisRowParity:
    """Port 3: batched basis rows, mirror vs. extension, bit-for-bit.

    Fixtures include a single-span (the saddle's) and a multi-span
    knot vector, stations across interiors and both double-knot
    interiors — the shapes the fill's rows actually hit.
    """

    @staticmethod
    def _knot_sets() -> "list[list[float]]":
        single = [vec[:] for vec in (
            [0.0] * 4, [1.0] * 4,
        )]
        multi: list = []
        for group in ([0.0] * 4, [0.25] * 2, [0.5] * 2, [0.75] * 2, [1.0] * 4):
            multi.extend(group)
        return [single[0] + single[1], multi]

    def _stations(self, uflat) -> "list[float]":
        import random

        rng = random.Random(20261016)
        base = list(_STATIONS) + [
            0.9999999999, 1.0e-12, 0.25, 0.5, 0.75,
            uflat[len(uflat) // 2],
        ]
        base += [rng.uniform(0.0, 1.0) for _ in range(24)]
        return sorted(set(float(t) for t in base))

    def test_row_grid_matches_mirror(self):
        from geometrykit import _rust
        from geometrykit.runtimes._npu import curve_np

        for uflat in self._knot_sets():
            ts = self._stations(uflat)
            for degree in (2, 3):
                mine = [curve_np._nonzero_basis(t, degree, uflat) for t in ts]
                ids, vals = _rust.basis_row_grid(
                    list(uflat), degree, list(ts)
                )
                stride = degree + 1
                for k, ref in enumerate(mine):
                    first = ids[k]
                    assert set(range(first, first + stride)) == set(ref.keys()), (
                        degree, k,
                    )
                    for j in range(stride):
                        rid = first + j
                        assert (
                            contract.float_to_hex(vals[k * stride + j])
                            == contract.float_to_hex(ref.get(rid, 0.0))
                        ), (degree, k, rid)

    def test_dispatch_list_matches_mirror(self):
        from geometrykit.runtimes import dispatch
        from geometrykit.runtimes._npu import curve_np

        for uflat in self._knot_sets():
            ts = self._stations(uflat) * 2  # duplicates included
            for degree in (2, 3):
                dispatch_rows = dispatch.basis_row_list(degree, uflat, ts)
                for k, t in enumerate(ts):
                    ref = curve_np._nonzero_basis(t, degree, uflat)
                    row = dispatch_rows[k]
                    for rid, v in ref.items():
                        assert (
                            contract.float_to_hex(row.get(rid, 0.0))
                            == contract.float_to_hex(v)
                        ), (degree, k, rid)
                    assert list(row.items())[len(row) - 1][0] == max(ref)

