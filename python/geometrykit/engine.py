# This file was generated with the assistance of an AI coding tool.
"""geometrykit engine probe.

Two engines implement one API against the contract:

- ``npu`` — the numpy/pure-Python mirror in ``runtimes/_npu/``; always
  available, the distribution's floor;
- ``rust`` — the ``geometrykit._rust`` extension module (maturin,
  ``crates/geometrykit-py``), an accelerator that never adds API.

Selection: explicit override wins, otherwise probe the extension.
``GEOMETRYKIT_ENGINE`` accepts ``npu``/``rust``/``auto``
(``auto`` = probe, the default).
"""

from __future__ import annotations

import os
from typing import Literal

Engine = Literal["npu", "rust"]
_OVERRIDE_KEYS = ("GEOMETRYKIT_ENGINE",)


def _probe_rust() -> bool:
    try:
        import geometrykit._rust  # noqa: F401
    except ImportError:
        return False
    return True


def engine() -> Engine:
    """Return the engine this process will use for kernel functions."""
    override = os.environ.get(_OVERRIDE_KEYS[0], "auto").strip().lower()
    if override in ("npu", "python", "mirror"):
        return "npu"
    has_rust = _probe_rust()
    if override == "rust":
        if not has_rust:
            raise RuntimeError(
                "GEOMETRYKIT_ENGINE=rust but the geometrykit._rust extension "
                "is not importable in this environment"
            )
        return "rust"
    return "rust" if has_rust else "npu"


def has_rust() -> bool:
    """True when the accelerator extension is importable."""
    return _probe_rust()
