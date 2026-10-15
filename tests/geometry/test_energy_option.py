# This file was generated with the assistance of an AI coding tool.
"""M6: the public `Surface.patch(fill=...)` routing, OCC-free throughout.

The whole continuity pipeline -- pure ribbons, support fields read off
kernel surfaces, plate fill, clamped slopes, G1+G2 rows -- runs in the
plain venv: no pythonocc, no numpy (adaptive stays off here). The twin
and the oracle comparisons live in the oracle suite; nothing here needs
them.
"""

from __future__ import annotations

import pytest
from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec
from geometrykit.geometry.surface import Surface
from geometrykit.runtimes._npu import patch_g1


def _network():
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


def _edge_cross(surface, edge: str, t: float):
    if edge == "south":
        return surface.tangents_at(t, 0.0)[1]
    if edge == "north":
        return surface.tangents_at(t, 1.0)[1]
    if edge == "west":
        return surface.tangents_at(0.0, t)[0]
    return surface.tangents_at(1.0, t)[0]


class TestFillRouting:
    """The flag selects the solver; bogus values fail loudly."""

    def test_coons_flag_gives_the_blend(self):
        curves = _network()
        surface = Surface.patch(curves, fill="coons", nu=8, nv=8)
        assert surface.nu > 0 and surface.nv > 0

    def test_energy_flag_gives_a_patch(self):
        curves = _network()
        supports = [
            Surface.ribbon(c, angle_deg=0, width=1.0, symmetric=True)
            for c in curves
        ]
        surface = Surface.patch(
            curves, constraints=["G1"] * 4, supports=supports,
            fill="energy", nu=12, nv=12,
        )
        assert surface.nu > 0 and surface.nv > 0

    def test_unknown_fill_raises(self):
        with pytest.raises(ValueError, match="unknown fill"):
            Surface.patch(_network(), fill="bogus")

    def test_energy_needs_four_curves(self):
        with pytest.raises(NotImplementedError):
            Surface.patch(_network()[:3], fill="energy")

    def test_g1_without_support_raises(self):
        with pytest.raises(ValueError, match="requires a support"):
            Surface.patch(
                _network(), constraints=["G1"] * 4, supports=[None] * 4,
                fill="energy",
            )


class TestEnergyRouteContinuity:
    """G1 through the public route, no OCC anywhere in the building."""

    def test_g1_holds_mid_edge(self):
        curves = _network()
        supports = [
            Surface.ribbon(c, angle_deg=0, width=1.0, symmetric=True)
            for c in curves
        ]
        surface = Surface.patch(
            curves, constraints=["G1"] * 4, supports=supports,
            fill="energy", tangency=1.0, g2=True, nu=12, nv=12,
        )
        towards = Vec(4, 3, 2.9)
        for edge, spine, support in zip(patch_g1.EDGE_NAMES, curves, supports):
            at = patch_g1.support_fields(support, spine, towards=towards)
            normal, _slope = at(0.5)
            d = _edge_cross(surface, edge, 0.5)
            residual = abs(d.x * normal[0] + d.y * normal[1] + d.z * normal[2])
            assert residual < 1e-9, (edge, residual)

    def test_corners_stay_exact(self):
        """Endpoints coincide; interior stations follow the station map
        (asserted exactly in the kernel suite)."""
        curves = _network()
        supports = [
            Surface.ribbon(c, angle_deg=0, width=1.0, symmetric=True)
            for c in curves
        ]
        surface = Surface.patch(
            curves, constraints=["G1"] * 4, supports=supports,
            fill="energy", nu=12, nv=12,
        )
        for edge, spine in zip(patch_g1.EDGE_NAMES, curves):
            # drawn direction may run either way along the edge, so each
            # patch corner must match *an* end of its spine
            ends = [spine.point_at(0.0), spine.point_at(1.0)]
            for t in (0.0, 1.0):
                got = _edge_cross_point(surface, edge, t)
                assert min(abs(got - e) for e in ends) < 1e-9, (edge, t)


def _edge_cross_point(surface, edge: str, t: float):
    if edge == "south":
        return surface.point_at(t, 0.0)
    if edge == "north":
        return surface.point_at(t, 1.0)
    if edge == "west":
        return surface.point_at(0.0, t)
    return surface.point_at(1.0, t)
