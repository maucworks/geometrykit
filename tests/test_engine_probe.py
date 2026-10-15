# This file was generated with the assistance of an AI coding tool.
"""Engine probe tests (plan §3: one API, two engines)."""

from __future__ import annotations

import pytest
from geometrykit import engine


def test_defaults_to_numpy_mirror():
    # the accelerator wheel is not part of this distribution yet (M6+)
    assert engine.engine() == "npu"
    assert engine.has_rust() is False


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
        with pytest.raises(RuntimeError, match="not importable"):
            engine.engine()
    finally:
        if old is None:
            del os.environ["GEOMETRYKIT_ENGINE"]
        else:
            os.environ["GEOMETRYKIT_ENGINE"] = old
