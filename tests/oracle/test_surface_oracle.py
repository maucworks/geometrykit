# This file was generated with the assistance of an AI coding tool.
"""Oracle layer: OCCT-anchored comparisons over the demo corpus (M2–M5 acceptance).

Runs only where pythonocc-core exists (ifckit's dev env). For every corpus
record the committed fixture carries the OCCT-derived expected numbers; the
kernel must reproduce them within the record's tolerance budget. The oracle
relationship is one-directional: OCCT is the incumbent, the kernel the
challenger — divergence is a test failure, never an API change.
"""

from __future__ import annotations

from ..conftest import load_corpus
from . import requires_occ


@requires_occ
class TestSurfaceOracle:
    """Surface.eval / normal round-trip against OCCT's Bezier/BSpline behaviour."""

    def test_corpus_loads(self, corpus):
        data = load_corpus(corpus)
        assert isinstance(data, dict)

    # M2: parameter-grid evaluation vs. occ_eval_point goldens per corpus record
    # M4: project_point (Newton) vs. occ_project_point goldens
    # M5: intersect_plane vs. occ_intersect_plane goldens + refit budgets
