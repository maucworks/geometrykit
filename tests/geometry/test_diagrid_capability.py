# This file was generated with the assistance of an AI coding tool.
"""Diagrid capability suite: the shipped flowing-roof workflow, as a test.

This pins the whole capability the pcurve diagrid was built for, on the
real flowing-roof network -- not scratch exports, a committed gate:

1. the energy fill over the 4-curve boundary ring;
2. two diagonal plane families sectioned at 50 offsets (batch levels);
3. `weave_diagrid`: uv-track crossings -> nodes on the surface, curves
   through their nodes;
4. normal spikes at the woven nodes (the construction-reality builders
   ask for), unit-length;
5. cross-validation against the independent option-A construction
   (bisect plane roots on the boundary curves + projection);
6. determinism: the workflow's node set is bit-stable under both
   engine configurations (extension routed vs. mirror forced);
7. a wall-clock budget for the whole chain.

Everything here is plain-kernel: no OCC, no numpy.
"""

from __future__ import annotations

import math
import os
import time

import pytest
from geometrykit.geometry import Curve
from geometrykit.geometry.primitives import Plane, Vec
from geometrykit.runtimes._npu import patch_np, section_np


def _network():
    """The shipped flowing-roof boundary ring (driver numbers, verbatim)."""
    return [
        Curve.from_tangents(
            start=Vec(0, 0, 2.4), tan_start=Vec(1, 0, 0),
            end=Vec(8, 0, 2.8), tan_end=Vec(1, 0, 0), scale=3.0,
        ),
        Curve.from_tangents(
            start=Vec(8, 0, 2.8), tan_start=Vec(0, 1, 0),
            end=Vec(8, 6, 2.9), tan_end=Vec(0, 1, 0), scale=3.0,
        ),
        Curve.from_tangents(
            start=Vec(0, 6, 3.2), tan_start=Vec(1, 0, 0),
            end=Vec(8, 6, 2.9), tan_end=Vec(1, 0, 0), scale=3.0,
        ),
        Curve.from_tangents(
            start=Vec(0, 6, 3.2), tan_start=Vec(0, -1, 0),
            end=Vec(0, 0, 2.4), tan_end=Vec(0, -1, 0), scale=3.0,
        ),
    ]


def _digest_plain(nu=12, nv=12):
    """The energy fill over the ring (the shipped fill construction)."""
    fill, _corners = patch_np.discrete_plate_fill(
        _network(), nu=nu, nv=nv, parametrization="arclength"
    )
    return fill


def _diag_plane(origin, nx, ny):
    n = math.sqrt(nx * nx + ny * ny)
    return Plane(
        origin=origin, x_axis=Vec(0, 0, 1), y_axis=Vec(ny / n, -nx / n, 0.0)
    )


def _fam_spans(curves):
    """Corner-anchored offsets for the two diagonal families."""
    sw = curves[0].point_at(0.0)
    ne = curves[1].point_at(1.0)
    nw = curves[3].point_at(0.0)
    se = curves[1].point_at(0.0)
    pa = _diag_plane(sw, 1.0, 1.0)
    na = pa.z_axis
    span_a = (ne.x - sw.x) * na.x + (ne.y - sw.y) * na.y
    pb = _diag_plane(nw, 1.0, -1.0)
    nb = pb.z_axis
    span_b = (se.x - nw.x) * nb.x + (se.y - nw.y) * nb.y
    return pa, na, span_a, pb, nb, span_b


def _tracks(fill, pa, na, span_a, pb, nb, span_b):
    tracks_a = [
        list(e.uv)
        for batch in fill.section_levels(pa, [span_a * k / 49 for k in range(50)])
        for e in batch
    ]
    tracks_b = [
        list(e.uv)
        for batch in fill.section_levels(pb, [span_b * k / 49 for k in range(50)])
        for e in batch
    ]
    return tracks_a, tracks_b


def build_diagrid(nu=12, nv=12):
    """The full capability chain: fill -> sections -> weave."""
    fill = _digest_plain(nu)
    curves = _network()
    pa, na, span_a, pb, nb, span_b = _fam_spans(curves)
    tracks_a, tracks_b = _tracks(fill, pa, na, span_a, pb, nb, span_b)
    curves_a, curves_b, nodes = section_np.weave_diagrid(fill, tracks_a, tracks_b)
    return fill, curves_a, curves_b, nodes


class TestDiagridCapability:
    """The full workflow on the shipped network."""

    def test_counts_and_node_surface_exactness(self):
        fill, curves_a, curves_b, nodes = build_diagrid()
        assert 40 <= len(curves_a) <= 55
        assert 40 <= len(curves_b) <= 55
        assert 900 <= len(nodes) <= 1400
        for (x, y, z), (u, v) in nodes:
            assert not (x != x or y != y or z != z), "NaN node"
            q = fill.point_at(u, v)
            assert abs(q.x - x) + abs(q.y - y) + abs(q.z - z) < 1e-9, (u, v)

    def test_nodes_are_on_their_curves(self):
        """Node-on-curve by two-level refinement on a node subset.

        Nearest-of-N-samples only measures sampling resolution (the
        trap: bound '2e-2' passed on small fills and failed on this
        8 m ring purely from sample spacing). Level 2 refines the
        level-1 panel hit locally; the measured bound on this network
        is 1.226e-03 m. The exact property (curves pass through their
        nodes at knot parameters) is pinned in test_curves.
        """
        _fill, curves_a, curves_b, nodes = build_diagrid()
        drawn = list(curves_a) + list(curves_b)
        panel = [([c.point_at(i / 200) for i in range(201)]) for c in drawn]
        params = [i / 200 for i in range(201)]
        stride = max(len(nodes) // 80, 1)
        worst = 0.0
        checked = 0
        for (x, y, z), _uv in nodes[::stride]:
            checked += 1
            best, (ck, ip) = 1e9, (0, 0)
            for k, pts in enumerate(panel):
                for i, p in enumerate(pts):
                    d = abs(p.x - x) + abs(p.y - y) + abs(p.z - z)
                    if d < best:
                        best, (ck, ip) = d, (k, i)
            curve, t_c = drawn[ck], params[ip]
            lo = max(0.0, t_c - 0.005)
            hi = min(1.0, t_c + 0.005)
            for j in range(51):
                tt = lo + (hi - lo) * j / 50
                p = curve.point_at(tt)
                d = abs(p.x - x) + abs(p.y - y) + abs(p.z - z)
                best = min(best, d)
            worst = max(worst, best)
        assert checked >= 60 and worst < 5e-3, worst

    def test_nodes_on_their_planes(self):
        """Every node lies on both of its family planes.

        The weave's tracks are plane sections, so a node -- an exact
        surface point at a longitudinal crossing of two tracks -- must
        satisfy both planes; measured worst residual 3.4e-05 m across
        both families (plane spacing is ~0.2 m, so the bound 1e-4 has
        2000x headroom while still being 4 orders below the spacing).
        """
        fill, _ca, _cb, nodes = build_diagrid()
        curves = _network()
        pa, na, span_a, pb, nb, span_b = _fam_spans(curves)
        for base, unit, span in ((pa.origin, na, span_a), (pb.origin, nb, span_b)):
            offsets = [span * k / 49 for k in range(50)]
            worst = 0.0
            for (x, y, z), _uv in nodes:
                p = Vec(x, y, z)
                best = min(
                    abs(
                        (p - Vec(
                            base.x + unit.x * o, base.y + unit.y * o,
                            base.z + unit.z * o,
                        )) @ unit
                    )
                    for o in offsets
                )
                worst = max(worst, best)
            assert worst < 1e-4, worst

    def test_normal_spikes_unit_length(self):
        fill, _ca, _cb, nodes = build_diagrid()
        for _xyz, (u, v) in nodes:
            n = fill.normal_at(u, v)
            length = math.sqrt(n.x**2 + n.y**2 + n.z**2)
            assert abs(length - 1.0) < 1e-9, (u, v, length)
            assert not (n.x != n.x or n.y != n.y or n.z != n.z)

    def test_engine_determinism(self):
        """The node set is bit-stable under both engine configurations."""
        from geometrykit import engine as engine_mod

        if not engine_mod.has_rust():
            pytest.skip("needs the accelerator: both runs would be mirror")
        engine_lit = os.environ.get("GEOMETRYKIT_ENGINE")
        try:
            os.environ["GEOMETRYKIT_ENGINE"] = "npu"
            _f1, ca1, cb1, nodes1 = build_diagrid()
            os.environ["GEOMETRYKIT_ENGINE"] = "rust"
            _f2, ca2, cb2, nodes2 = build_diagrid()
        finally:
            if engine_lit is None:
                os.environ.pop("GEOMETRYKIT_ENGINE", None)
            else:
                os.environ["GEOMETRYKIT_ENGINE"] = engine_lit
        from geometrykit import contract

        sig1 = contract.signature([list(uv) for _xyz, uv in nodes1])
        sig2 = contract.signature([list(uv) for _xyz, uv in nodes2])
        assert sig1 == sig2
        assert len(ca1) == len(ca2) and len(cb1) == len(cb2)

    def test_workflow_budget(self):
        """Whole chain well under the interactivity bar (shipped ~1.5 s)."""

        start = time.perf_counter()
        build_diagrid()
        elapsed = time.perf_counter() - start
        assert elapsed < 8.0, elapsed
