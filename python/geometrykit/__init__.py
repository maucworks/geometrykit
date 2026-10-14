# This file was generated with the assistance of an AI coding tool.
"""geometrykit — geometry primitives and NURBS data model.

v0.1: first API release (contract v0, Tier-1 promotion of the geometry
primitives from ifckit, engine probe, and the M-phase test corpus).
"""

from geometrykit import contract, engine
from geometrykit.geometry import (
    Arc,
    Curve,
    Field,
    HexPattern,
    Line,
    Path,
    Plane,
    Polyline,
    Spiral,
    Surface,
    Transform,
    Vec,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "contract",
    "engine",
    # geometry (value types — guide tier)
    "Vec",
    "Plane",
    "Line",
    "Arc",
    "Polyline",
    "Path",
    "Curve",
    "Spiral",
    "Surface",
    "Transform",
    "Field",
    "HexPattern",
]
