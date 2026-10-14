# This file was generated with the assistance of an AI coding tool.
"""Symmetric support ribbons: the spine in the middle, as a tangency support.

The CATIA habit this follows: a ribbon is a *support*, not a flange. Building
it symmetric -- the spine down the middle -- makes the constraint independent
of which side the patch approaches from, which matters where one ribbon serves
two patches (the flowing-roof ridge).

Two things are asserted here:
- the spine lies *exactly* on the ribbon's middle line (so the support really
  carries the constraint curve), and
- the constraint plane is the one the curve's own cross direction defines --
  identical to a one-sided ribbon's, which is why switching does not move the
  geometry.

The one-sided case is kept: OCC's MakeFilling needs the constraint edge to be
an edge of the support face, which a symmetric ribbon only satisfies once the
face is bounded at the spine (see `Surface.patch`).
"""

from __future__ import annotations

import pytest
from geometrykit.geometry.curve import Curve
from geometrykit.geometry.primitives import Vec
from geometrykit.geometry.surface import Surface
from geometrykit.runtimes._npu import patch_g1, patch_np


def _occ_available() -> bool:
    try:
        import OCC.Core.BRep  # noqa: F401
        return True
    except ImportError:
        return False


def _spine() -> Curve:
    return Curve.from_tangents(
        start=Vec(0, 0, 2.5), tan_start=Vec(1, 0, 0),
        end=Vec(8, 0, 2.7), tan_end=Vec(1, 0, 0), scale=3.0,
    )


class TestSpineFieldFromSupport:
    """The kernel side: a normal field read off a support, occ-free."""

    @staticmethod
    def _flat_strip(spine, shift):
        """A translation strip: *spine*, and the same control net moved by *shift*."""
        from geometrykit.geometry.curve import Curve as _Curve

        moved = _Curve(
            control_points=[p + shift for p in spine.points],
            knots=list(spine.knots),
            multiplicities=list(spine.multiplicities),
            degree=spine.degree,
        )
        return Surface(
            control_points=[
                [spine.points[i], moved.points[i]] for i in range(len(spine.points))
            ],
            uknots=list(spine.knots),
            vknots=[0.0, 1.0],
            umults=list(spine.multiplicities),
            vmults=[2, 2],
            udegree=spine.degree,
            vdegree=1,
        )

    def test_field_is_the_support_normal_on_the_spine(self):
        """The field is the strip's normal there: unit, and off the strip's span."""
        spine = _spine()
        strip = self._flat_strip(spine, Vec(0, -1, 0))
        field = patch_g1.normal_field_from_support(strip, spine)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            nx, ny, nz = field(t)
            assert abs(nx * nx + ny * ny + nz * nz - 1.0) < 1e-9, t
            # perpendicular to the spine tangent and to the strip's span
            tangent = spine.tangent_at(t)
            length = (tangent.x**2 + tangent.y**2 + tangent.z**2) ** 0.5
            along = (nx * tangent.x + ny * tangent.y + nz * tangent.z) / length
            assert abs(along) < 1e-6, (t, along)
            assert abs(ny) < 1e-9, (t, ny)

    def test_field_raises_when_the_spine_misses_the_support(self):
        spine = _spine()
        far = Curve.from_tangents(
            start=Vec(0, 50, 2.5), tan_start=Vec(1, 0, 0),
            end=Vec(8, 50, 2.7), tan_end=Vec(1, 0, 0), scale=3.0,
        )
        strip = self._flat_strip(far, Vec(0, -1, 0))
        with pytest.raises(patch_g1.G1Error, match="misses its support"):
            patch_g1.normal_field_from_support(strip, spine)(0.5)


@pytest.mark.skipif(not _occ_available(), reason="needs pythonocc for ribbons")
class TestSymmetricRibbon:
    """The support side: a symmetric ribbon carries the spine exactly."""

    def test_spine_lies_on_the_middle_line(self):
        spine = _spine()
        ribbon = Surface.ribbon(spine, angle_deg=0, width=1.0, symmetric=True)
        worst = 0.0
        for k in range(21):
            t = k / 20.0
            worst = max(worst, abs(ribbon.point_at(t, 0.5) - spine.point_at(t)))
        assert worst < 1e-9, worst

    def test_projection_finds_the_spine_in_the_middle(self):
        spine = _spine()
        ribbon = Surface.ribbon(spine, angle_deg=0, width=1.0, symmetric=True)
        for t in (0.1, 0.5, 0.9):
            point = spine.point_at(t)
            projected, (_u, v) = ribbon.project(point)
            assert abs(projected - point) < 1e-6
            assert abs(v - 0.5) < 1e-6, (t, v)

    def test_constraint_plane_matches_the_one_sided_ribbon(self):
        """Switching to symmetric must not move the tangent plane."""
        spine = _spine()
        sided = Surface.ribbon(spine, angle_deg=0, width=1.0)
        symmetric = Surface.ribbon(spine, angle_deg=0, width=1.0, symmetric=True)
        sided_field = patch_g1.normal_field_from_support(sided, spine)
        symmetric_field = patch_g1.normal_field_from_support(symmetric, spine)
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            a, b = sided_field(t), symmetric_field(t)
            dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
            assert abs(abs(dot) - 1.0) < 1e-6, (t, dot)

    def test_patch_accepts_a_symmetric_support(self):
        """OCC's filler needs the spine to be an edge of the support face."""
        network = [
            Curve.from_tangents(
                start=Vec(0, 0, 2.5), tan_start=Vec(1, 0, 0),
                end=Vec(8, 0, 2.7), tan_end=Vec(1, 0, 0), scale=3.0,
            ),
            Curve.from_tangents(
                start=Vec(8, 0, 2.7), tan_start=Vec(0, 1, 0),
                end=Vec(8, 6, 3.1), tan_end=Vec(0, 1, 0), scale=3.0,
            ),
            Curve.from_tangents(
                start=Vec(8, 6, 3.1), tan_start=Vec(-1, 0, 0),
                end=Vec(0, 6, 2.9), tan_end=Vec(-1, 0, 0), scale=3.0,
            ),
            Curve.from_tangents(
                start=Vec(0, 6, 2.9), tan_start=Vec(0, -1, 0),
                end=Vec(0, 0, 2.5), tan_end=Vec(0, -1, 0), scale=3.0,
            ),
        ]
        supports = [
            Surface.ribbon(curve, angle_deg=0, width=1.0, symmetric=True)
            for curve in network
        ]
        surface = Surface.patch(
            curves=network,
            constraints=["G1"] * 4,
            supports=supports,
            tol=1e-4,
        )
        assert surface is not None and surface.nu > 0

    def test_kernel_patch_solves_against_symmetric_supports(self):
        """Both patches of the roof, with the shared ridge ribbon symmetric."""
        network = [
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
        ridge = Surface.ribbon(
            network[2], angles_deg=[180, 155, 180], width=1.0, symmetric=True
        )
        supports = [
            Surface.ribbon(network[0], angle_deg=0, width=1.0, symmetric=True),
            Surface.ribbon(network[1], angle_deg=0, width=1.0, symmetric=True),
            ridge,
            Surface.ribbon(network[3], angle_deg=0, width=1.0, symmetric=True),
        ]
        params_u, params_v, oriented = patch_np.coons_parametrization(network, nu=12, nv=12)
        fields = {}
        for edge, spine, support in zip(patch_g1.EDGE_NAMES, network, supports):
            patch_params = params_u if edge in ("south", "north") else params_v
            curve_params = patch_np.edge_curve_params(oriented, edge, patch_params)
            on_spine = patch_g1.normal_field_from_support(support, spine)
            fields[edge] = (
                lambda t, cp=curve_params, f=on_spine: f(
                    min(cp, key=lambda c: abs(c - t))
                )
            )
        surface, _corners, stations = patch_g1.constrained_coons_patch(
            network, fields, nu=12, nv=12
        )
        for edge in patch_g1.EDGE_NAMES:
            params = stations[0] if edge in ("south", "north") else stations[1]
            worst = 0.0
            for t in params[2:-2]:
                if edge == "south":
                    d = surface.tangents_at(t, 0.0)[1]
                elif edge == "north":
                    d = surface.tangents_at(t, 1.0)[1]
                elif edge == "west":
                    d = surface.tangents_at(0.0, t)[0]
                else:
                    d = surface.tangents_at(1.0, t)[0]
                n = fields[edge](t)
                worst = max(worst, abs(d.x * n[0] + d.y * n[1] + d.z * n[2]))
            assert worst < 1e-6, (edge, worst)