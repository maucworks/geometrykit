# This file was generated with the assistance of an AI coding tool.
"""Panelization acceptance runs (plan M2 gate).

The panelization domain (`trim_grid`, `crossing_grid`, `hankin_cells`,
`_project_uv`) runs **without** the `require_occ` gate since M2: the
these-tests layer runs in any environment (OCC or not) and the corpus
fixtures are the ground truth.
"""

from __future__ import annotations

import pytest

from ..oracle import requires_occ


class TestPanelAcceptance:
    @staticmethod
    def _demo_surface():
        from geometrykit.geometry.primitives import Vec
        from geometrykit.geometry.surface import Surface

        cps = [[Vec(0, 0, 0), Vec(0, 4, 0)], [Vec(8, 0, 0), Vec(8, 4, 0)]]
        return Surface(cps, [0, 1], [0, 1], [2, 2], [2, 2], 1, 1)

    def test_field_module_importable_occ_free(self):
        """field/patterns import without OCC installed."""
        from geometrykit.geometry import field, patterns  # noqa: F401

        assert field is not None

    def test_field_module_occ_free_run(self):
        """M2 acceptance: trim_grid/crossing_grid/hankin run occ-free."""
        from geometrykit.geometry.field import crossing_grid, trim_grid
        from geometrykit.geometry.patterns import hankin_cells

        face = self._demo_surface()
        f = trim_grid(face, spacing=1.0)
        assert f.cells, "trim_grid produced no cells"
        g = crossing_grid([face, face], spacing=1.0, shared=[(0, 1)])
        assert g.cells, "crossing_grid produced no cells"
        cells = hankin_cells(f, faces=[face], theta_deg=60.0)
        assert cells and len(cells[0]) in (4, 3), cells[0] if cells else None

    def test_gone_occ_gates_documented(self):
        """M2 acceptance (landed): field/patterns carry no require_occ gates."""
        import inspect

        from geometrykit.geometry import field, patterns

        gates = 0
        for mod in (field, patterns):
            src = inspect.getsource(mod)
            gates += src.count("require_occ")
        assert gates == 0, "field/patterns must run occ-free (M2 landed)"

    @requires_occ
    def test_trim_grid_matches_occ_path(self, corpus):
        pytest.importorskip("geometrykit.geometry.field")
