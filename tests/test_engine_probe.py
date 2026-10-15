# This file was generated with the assistance of an AI coding tool.
"""Engine probe tests (plan §3: one API, two engines)."""

from __future__ import annotations

import pytest
from geometrykit import engine


def test_defaults_to_probed_engine():
    import os

    # no explicit override: the extension decides. Both outcomes are
    # valid scaffold states — mirror-only venvs and accelerator-built ones.
    override = os.environ.get("GEOMETRYKIT_ENGINE", "auto").strip().lower()
    if override in ("npu", "python", "mirror"):
        assert engine.engine() == "npu"
    elif override == "rust":
        if engine.has_rust():
            assert engine.engine() == "rust"
    else:
        assert engine.engine() == ("rust" if engine.has_rust() else "npu")


def test_explicit_npu_override():
    engine.engine  # attr exists
    import os

    old = os.environ.get("GEOMETRYKIT_ENGINE")
    os.environ["GEOMETRYKIT_ENGINE"] = "npu"
    try:
        assert engine.engine() == "npu"
    finally:
        if old is None:
            del os.environ["GEOMETRYKIT_ENGINE"]
        else:
            os.environ["GEOMETRYKIT_ENGINE"] = old


def test_rust_override_without_extension_raises():
    import os

    old = os.environ.get("GEOMETRYKIT_ENGINE")
    os.environ["GEOMETRYKIT_ENGINE"] = "rust"
    try:
        if engine.has_rust():
            assert engine.engine() == "rust"
        else:
            with pytest.raises(RuntimeError, match="not importable"):
                engine.engine()
    finally:
        if old is None:
            del os.environ["GEOMETRYKIT_ENGINE"]
        else:
            os.environ["GEOMETRYKIT_ENGINE"] = old
