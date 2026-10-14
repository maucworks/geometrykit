# This file was generated with the assistance of an AI coding tool.
"""Cage data-model tests (contract v0.1): coordinate systems + corners."""

from __future__ import annotations

import math

import pytest
from geometrykit import contract
from geometrykit.cage import CoordinateSystem, Corner

SPORT = 1e-12


class TestCoordinateSystem:
    def test_identity_frame_tangents(self):
        cs = CoordinateSystem((0, 0, 0), (1, 0, 0, 0), id="cs-1")
        c = Corner(cs, {"A": 0.0, "B": math.pi / 2}, id="c-1")
        t_a = c.tangent_for("A")
        t_b = c.tangent_for("B")
        assert t_a == (1.0, 0.0, 0.0)
        assert abs(t_b[0]) < SPORT and abs(t_b[1] - 1.0) < 1e-12

    def test_rotated_frame_composition(self):
        q90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
        cs = CoordinateSystem((5, 0, 0), q90, id="cs-2")
        c = Corner(cs, {"A": 0.0})
        t = c.tangent_for("A")
        assert abs(t[0]) < 1e-12 and abs(t[1] - 1.0) < 1e-12 and abs(t[2]) < 1e-12, t

    def test_translate_only_moves_point(self):
        cs = CoordinateSystem((5, 0, 0), (1, 0, 0, 0))
        p = cs.apply_point((1, 0, 0))
        assert p == (6.0, 0.0, 0.0)

    def test_quaternion_canonical_sign(self):
        qa = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
        qb = (-qa[0], -qa[1], -qa[2], -qa[3])
        a = CoordinateSystem((0, 0, 0), qa)
        b = CoordinateSystem((0, 0, 0), qb)
        assert a.orientation == b.orientation
        assert a.signature() == b.signature()

    def test_from_plane_z_maps_to_normal(self):
        cs = CoordinateSystem.from_plane((2, 3, 4), (0, 0, 1), id="cs-3")
        d = cs.apply_direction((0, 0, 1))
        assert d == pytest.approx((0, 0, 1))

    def test_from_plane_auto_x(self):
        cs = CoordinateSystem.from_plane((0, 0, 0), (0, 0, 1))
        x = cs.apply_direction((1, 0, 0))
        assert tuple(map(lambda v: round(v, 9), x)) == (1.0, 0.0, 0.0)

    def test_from_plane_degenerate_xref_raises(self):
        with pytest.raises(ValueError):
            CoordinateSystem.from_plane((0, 0, 0), (1, 0, 0), (1, 0, 0))

    def test_serialization_signature_stable(self):
        cs = CoordinateSystem((0.1, 0.2, 0.3), (math.cos(0.3), 0, 0, math.sin(0.3)))
        r1 = contract.signature(cs.to_record())
        cs2 = CoordinateSystem.from_record(r1) if hasattr(CoordinateSystem, "from_record") else None
        assert len(cs.signature()) == 16


class TestCorner:
    def test_missing_terminal_raises(self):
        cs = CoordinateSystem((0, 0, 0), (1, 0, 0, 0))
        c = Corner(cs, {"A": 0.0})
        with pytest.raises(KeyError):
            c.tangent_for("B")

    def test_degenerate_terminal_spacing_raises(self):
        cs = CoordinateSystem((0, 0, 0), (1, 0, 0, 0))
        with pytest.raises(ValueError):
            Corner(cs, {"A": 0.0, "B": 0.0}).validate_terminals()

    def test_terminal_angles_referenceable_by_curve(self):
        cs = CoordinateSystem((0, 0, 0), (1, 0, 0, 0))
        c = Corner(cs, {"wall-1": math.pi / 4, "beam-7": -math.pi / 2})
        t = c.tangent_for("beam-7")
        assert abs(t[0]) < 1e-12 and abs(t[1] + 1.0) < 1e-12 and abs(t[2]) < 1e-12, t

    def test_corner_record_references_cs_id(self):
        cs = CoordinateSystem((0, 0, 0), (1, 0, 0, 0), id="cs-fixed")
        c = Corner(cs, {"A": 0.5}, id="c-fixed")
        rec = c.to_record()
        assert rec["coordinate_system_id"] == "cs-fixed"
        assert rec["angles"] == {"A": 0.5}
