"""Optional IFC bridge for geometrykit.

geometrykit is deliberately IFC-free (IFC serialisation stays in ifckit).
A host application that wants the
legacy ``to_ifc_*`` serialisation methods on geometry objects installs the
ifckit IFC helper module as a bridge — the object must expose the helpers
``geometrykit`` callers use today: ``pt3`` (cartesian point) and
``directrix_from_path``/``directrix_from_line``/``directrix_from_arc``
(composite-curve directrix builders), matching the surface of
``ifckit.geometry.ifc_geom``.
"""

from __future__ import annotations

from typing import Any, Protocol


class IfcBridge(Protocol):
    """Structural protocol for the host-side IFC bridge."""

    def pt3(self, ifc_file: Any, x: float, y: float, z: float) -> Any:
        ...


_bridge: Any | None = None


def install_ifc_bridge(bridge: Any) -> None:
    """Install the host-side IFC bridge (idempotent)."""
    global _bridge
    _bridge = bridge


def uninstall_ifc_bridge() -> None:
    """Drop the bridge (used by tests)."""
    global _bridge
    _bridge = None


def require_bridge() -> Any:
    """Return the installed bridge or raise with clear instructions."""
    if _bridge is None:
        raise RuntimeError(
            "geometrykit: an IFC serialisation was requested but no IFC bridge "
            "is installed. Import ifckit (which installs the bridge) before "
            "calling to_ifc_* methods, or avoid IFC serialisation here."
        )
    return _bridge
