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
                            == contract.float_to_hex(ref[rid])
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


@pytest.mark.rust
@pytest.mark.skipif(not engine.has_rust(), reason="requires geometrykit._rust")
class TestInterpolationParity:
    """Port 6: global interpolation, mirror vs. extension, bit-for-bit."""

    @staticmethod
    def _point_sets() -> "list[list[_Vec]]":
        import math as _m
        import random as _r

        rng = _r.Random(20261017)
        sine = [
            _Vec(k * 0.2, _m.sin(k * 0.5) * (1.0 if k % 2 else 0.7), 0.0)
            for k in range(50)
        ]
        kinked = [
            _Vec(k, k if k < 3 else 6.0 - k, 0.0) for k in range(6)
        ]
        hammered = [
            _Vec(rng.uniform(-1, 1) * 3.0, rng.uniform(-1, 1) * 2.0, rng.uniform(-1, 1))
            for _ in range(40)
        ]
        return [sine, kinked, hammered]

    @staticmethod
    def _compare(points, **kwargs):
        from geometrykit.runtimes import dispatch
        from geometrykit.runtimes._npu import curve_np

        c_mirror, k_mirror, m_mirror = curve_np.interpolate(
            points, **kwargs
        )
        c_rust, k_rust, m_rust = dispatch.interpolate(points, **kwargs)
        assert k_rust == k_mirror and m_rust == m_mirror
        for a, b in zip(c_mirror, c_rust):
            for va, vb in zip(a, b):
                assert contract.float_to_hex(va) == contract.float_to_hex(vb), (a, b)

    def test_free_and_tangent_cases(self):
        for pts_ in self._point_sets():
            self._compare(pts_)
            self._compare(
                pts_, tan_start=_Vec(1, 0, 0), tan_end=_Vec(1, 0, 0)
            )
            self._compare(pts_, degree=1)

    def test_explicit_params(self):
        import math

        pts_ = [_Vec(k, math.sin(k * 0.7), 0.0) for k in range(24)]
        total = 0.0
        params = [0.0]
        for a, b in zip(pts_, pts_[1:]):
            total += abs(a - b)
            params.append(total)
        params = [p / total for p in params]
        self._compare(pts_, knots_param=params)
        self._compare(pts_, knots_param=params, degree=2)

    def test_singular_has_the_error(self):
        from geometrykit.runtimes import dispatch
        from geometrykit.runtimes._npu import curve_np

        # identical points collapse chord params to all-zeros: the
        # elimination has one pivot column with no pivot (mirror:
        # RuntimeError; the extension maps core's Singular kind to the
        # same type)
        bad = [_Vec(0.0, 0.0, 0.0)] * 3
        for fn in (curve_np.interpolate, dispatch.interpolate):
            with pytest.raises(RuntimeError, match="singular"):
                fn(bad, degree=3)


@pytest.mark.rust
@pytest.mark.skipif(not engine.has_rust(), reason="requires geometrykit._rust")
class TestSectionParity:
    """Port 5: level-set sections, mirror vs. extension, bit-for-bit.

    The saddle fixture with offset sweeps spanning [empty ... grazing
    ... mid ... empty]: every chain's uv track AND evaluated 3D points
    compared bitwise; empty offsets empty in both engines.
    """

    def test_single_plane_spread(self):
        from geometrykit.runtimes import dispatch
        from geometrykit.runtimes._npu import section_np

        surf = _surface()
        normal = (1.0, 0.3, 0.4)
        for off in (-0.5, 0.0, 0.5):
            origin = (0.5 + off * normal[0], 0.5 + off * normal[1], 2.6 + off * normal[2])
            mine = dispatch.plane_section_chains(surf, list(origin), list(normal), [0.0], 12, 12)[0]
            ref = section_np.section_plane(surf, origin, normal, nu=12, nv=12)
            assert len(mine) == len(ref), off
            for (m_pts, m_uvs), (r_pts, r_uvs) in zip(mine, ref):
                assert len(m_pts) == len(r_pts), off
                for (pm, um), (pr, ur) in zip(zip(m_pts, m_uvs), zip(r_pts, r_uvs)):
                    for a, b in (
                        (pm[0], pr[0]), (pm[1], pr[1]), (pm[2], pr[2]),
                        (um[0], ur[0]), (um[1], ur[1]),
                    ):
                        assert (
                            contract.float_to_hex(a) == contract.float_to_hex(b)
                        ), (off, a, b)

    def test_levels_bitwise_vs_mirror(self):
        from geometrykit.runtimes import dispatch
        from geometrykit.runtimes._npu import section_np

        surf = _surface()
        origin = (0.5, 0.5, 2.6)
        normal = (1.0, 0.3, 0.4)
        offsets = [-1.0, -0.2, 0.0, 0.2, 0.6, 1.0]
        mine = dispatch.plane_section_chains(
            surf, list(origin), list(normal), offsets, 48, 48
        )
        ref = section_np.section_levels(
            surf, origin, normal, offsets, nu=48, nv=48
        )
        assert len(mine) == len(ref) == len(offsets)
        for chains_m, chains_r, off in zip(mine, ref, offsets):
            same_empty = (not chains_m) or (not chains_r)
            if same_empty:
                assert len(chains_m) == len(chains_r), (off, chains_m, chains_r)
                continue
            assert len(chains_m) == len(chains_r), off
            for (m_pts, m_uvs), (r_pts, r_uvs) in zip(chains_m, chains_r):
                assert len(m_pts) == len(r_pts), off
                for (pm, um), (pr, ur) in zip(
                    zip(m_pts, m_uvs), zip(r_pts, r_uvs)
                ):
                    for a, b in ((pm[0], pr[0]), (pm[1], pr[1]), (pm[2], pr[2]),
                                 (um[0], ur[0]), (um[1], ur[1])):
                        assert (
                            contract.float_to_hex(a) == contract.float_to_hex(b)
                        ), (off, a, b)

@pytest.mark.rust
@pytest.mark.skipif(not engine.has_rust(), reason="requires geometrykit._rust")
class TestBandedParity:
    """Port 4: banded Cholesky + solve, mirror vs. extension, bit-for-bit."""

    @staticmethod
    def _rows(n: int) -> list:
        rows = [{} for _ in range(n)]
        for i in range(n):
            rows[i][i] = rows[i].get(i, 0.0) + 2.0
            if i + 1 < n:
                rows[i].setdefault(i, 0.0)
            if i + 1 < n:
                rows[i][i + 1] = rows[i].get(i + 1, 0.0) - 1.0
                rows[i + 1][i] = rows[i + 1].get(i, 0.0) - 1.0
        return rows

    def test_factor_and_solve_match_mirror(self):
        from geometrykit.runtimes import dispatch
        from geometrykit.runtimes._npu import linalg_np

        n = 24
        rows = self._rows(n)
        bands = 2
        f_mirror = linalg_np.banded_cholesky(rows, n, bands)
        f_rust = dispatch.banded_cholesky(rows, n, bands)
        for i in range(n):
            for k in range(bands + 1):
                assert (
                    contract.float_to_hex(f_rust[i][k])
                    == contract.float_to_hex(f_mirror[i][k])
                ), (i, k)
        rhs = [float(k % 5) - 2.0 for k in range(n)]
        x_mirror = linalg_np.banded_solve(f_mirror, rhs, n, bands)
        x_rust = dispatch.banded_solve(f_rust, rhs, n, bands)
        for a, b in zip(x_mirror, x_rust):
            assert contract.float_to_hex(a) == contract.float_to_hex(b)

    def test_non_spd_raises_like_the_mirror(self):
        from geometrykit.runtimes import dispatch
        from geometrykit.runtimes._npu import linalg_np

        rows = [{0: 0.0}]
        with pytest.raises(ValueError, match="not positive definite"):
            linalg_np.banded_cholesky(rows, 1, 1)
        with pytest.raises(ValueError, match="not positive definite"):
            dispatch.banded_cholesky(rows, 1, 1)

    def test_band_overflow_raises_like_the_mirror(self):
        from geometrykit.runtimes import dispatch
        from geometrykit.runtimes._npu import linalg_np

        rows = [{0: 2.0, 9: 1.0}]
        for fn in (linalg_np.banded_cholesky, dispatch.banded_cholesky):
            with pytest.raises(ValueError, match="beyond the bandwidth"):
                fn(rows, 3, 1)

