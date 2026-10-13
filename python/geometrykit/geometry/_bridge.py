"""Optional IFC bridge for geometrykit.

geometrykit is deliberately IFC-free (see ``plans/geometrykit/PLAN-occ-exit.md``
§3: IFC serialisation stays in ifckit). A host application that wants the
legacy ``to_ifc_*`` serialisation methods on geometry objects installs a
bridge object exposing the small helper surface ``ifckit.geometry.ifc_geom``
provides today:

- ``pt3(ifc_file, x, y, z)`` → an ``IfcCartesianPoint`` helper

ifckit installs its bridge when its geometry facade is imported
(``install_ifc_bridge``). Everything works without the bridge as long as no
``to_ifc_*`` method is called; calling one without a bridge raises a
``RuntimeError`` with instructions instead of an ``ImportError`` mid-import
(often swallowed and re-raised confusingly).
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
