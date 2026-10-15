# This file was generated with the assistance of an AI coding tool.
"""Oracle layer: OCCT-anchored kernel comparisons (M2 acceptance).

Runs only where pythonocc-core exists (ifckit's dev conda env). The kernel
must reproduce OCCT within tolerance for evaluation, D1/normals, and
projection on the committed test surfaces. OCCT is the incumbent authority,
never a dependency of geometrykit itself.
"""

from __future__ import annotations

import math

import pytest
from geometrykit.geometry.primitives import Vec
from geometrykit.geometry.surface import (
    Surface,
    occ_eval_point,
    occ_normal_at,
    occ_project_point,
)

from . import requires_occ


def _bilinear() -> Surface:
    cps = [[Vec(0, 0, 0), Vec(0, 4, 0)], [Vec(8, 0, 0), Vec(8, 4, 0)]]
    return Surface(cps, [0, 1], [0, 1], [2, 2], [2, 2], 1, 1)


def _rational_quarter_cylinder() -> Surface:
    r_c, h = 2.0, 5.0
    w = 1.0 + math.sqrt(2) / 2
    mcps = [
        [Vec(r_c, 0, 0), Vec(r_c, 0, h)],
        [Vec(r_c, r_c, 0), Vec(r_c, r_c, h)],
        [Vec(0, r_c, 0), Vec(0, r_c, h)],
    ]
    ws = [[1.0, 1.0], [w, w], [1.0, 1.0]]
    return Surface(mcps, [0, 1], [0, 1], [3, 3], [2, 2], 2, 1, weights=ws)


@pytest.mark.oracle
@requires_occ
class TestEvalKernelOracle:
    @pytest.mark.parametrize("surface", [_bilinear(), _rational_quarter_cylinder()])
    def test_eval_matches_occ(self, surface):
        checks = 0
        for i in range(9):
            for j in range(9):
                u, v = i / 8, j / 8
                a = surface.point_at(u, v)
                b = occ_eval_point(surface, u, v)
                assert abs(a.x - b.x) < 1e-9 and abs(a.y - b.y) < 1e-9 and abs(a.z - b.z) < 1e-9
                checks += 1
        assert checks == 81

    @pytest.mark.parametrize("surface", [_bilinear(), _rational_quarter_cylinder()])
    def test_normal_matches_occ(self, surface):
        for i in range(5):
            for j in range(5):
                u, v = i / 4, j / 4
                a = surface.normal_at(u, v)
                b = occ_normal_at(surface, u, v)
                dot = a.x * b.x + a.y * b.y + a.z * b.z
                assert abs(abs(dot) - 1.0) < 1e-9, (u, v, a, b)

    @pytest.mark.parametrize("surface", [_bilinear(), _rational_quarter_cylinder()])
    def test_projection_matches_occ(self, surface):
        # on-surface probes: the kernel uv must equal the OCC projection uv
        for i in range(5):
            for j in range(5):
                u, v = i / 4, j / 4
                pt = surface.point_at(u, v)
                a_uv = surface.project(pt)[1]
                _, b_uv = occ_project_point(surface, pt)
                assert abs(a_uv[0] - b_uv[0]) < 1e-6, (u, v, a_uv, b_uv)
                assert abs(a_uv[1] - b_uv[1]) < 1e-6, (u, v, a_uv, b_uv)

    def test_off_surface_projection_agrees(self):
        s = _rational_quarter_cylinder()
        probes = [Vec(2.8, 1.1, 2.0), Vec(0.4, 1.8, 4.0), Vec(2.0, 0.5, 1.0)]
        for pt in probes:
            a = s.project(pt)[1]
            _, b_uv = occ_project_point(s, pt)
            distance_kernel = s.point_at(*a)
            d1 = (distance_kernel.x - pt.x) ** 2 + (distance_kernel.y - pt.y) ** 2 + (
                distance_kernel.z - pt.z
            ) ** 2
            bo = occ_eval_point(s, *b_uv)
            d2 = (bo.x - pt.x) ** 2 + (bo.y - pt.y) ** 2 + (bo.z - pt.z) ** 2
            # equal minima within solver slop (uv may differ on flat sheets)
            assert abs(d1 - d2) < 1e-9, (pt, d1, d2)
