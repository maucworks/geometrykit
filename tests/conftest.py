# This file was generated with the assistance of an AI coding tool.
"""Shared fixtures: corpus loading + engine selection.

Corpus fixtures are committed JSON snapshots captured from the A320
configurator demo corpus (``scripts/capture_corpus.py``, run where ifckit
+ deps exist). Each record follows geometrykit contract v0: `{id, kind,
body, sig}`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Suite runs from any interpreter (any env with pythonocc included) without an
# editable install: put the package dir (repo root /python) on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG = _REPO_ROOT / "python"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import geometrykit  # noqa: E402  (path bootstrap happens above)

FIXTURES = Path(__file__).parent / "fixtures"


def corpus_names() -> list[str]:
    files = sorted(FIXTURES.glob("corpus/*.json"))
    return [f.stem for f in files]


@pytest.fixture(scope="session")
def corpus(request):
    """Parametrize over all committed corpus snapshots."""
    return request.param


def pytest_generate_tests(metafunc):
    if "corpus" in metafunc.fixturenames:
        names = corpus_names()
        metafunc.parametrize("corpus", names, indirect=True, ids=names)


def load_corpus(name: str) -> dict:
    return json.loads((FIXTURES / "corpus" / f"{name}.json").read_text())


@pytest.fixture(scope="session")
def gk():
    """The geometrykit public surface under test."""
    return geometrykit
