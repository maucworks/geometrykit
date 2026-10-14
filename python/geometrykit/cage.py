# This file was generated with the assistance of an AI coding tool.
"""Spline-cage data records: coordinate systems + corners.

Contract v0.1 additions (plan: OCC-exit; cage data-model sessions):

- ``CoordinateSystem`` — a placed Cartesian frame in model space
  (position + quaternion). Generic; carries no cage semantics itself.
- ``Corner`` — the cage composition: binds a ``CoordinateSystem`` to
  curve terminals via a dict of angles (radians, XY polar around the
  frame's +z). The **XY plane is the planarity plane** for cage
  tangents (in-plane tangents have polar z = 0).

Conventions (pinned, see also ``contract.py``):

- quaternion order ``(qw, qx, qy, qz)``, Hamilton, right-handed,
  normalized, **canonical sign ``qw ≥ 0``** (q and −q are the same
  rotation; signature stability requires a single representation);
- ``tangent_for`` returns the tangent in **3D model space** (the frame's
  local XY polar direction at θ, mapped through the quaternion);
- serialization is contract-canonical (hex-float bits) — cross-language
  stable, same hashing rules as everything else.
"""

from __future__ import annotations

import math
from typing import Optional

from geometrykit import contract as _contract


def _normalize_quaternion(qw: float, qx: float, qy: float, qz: float) -> tuple:
    """Normalize and canonically sign-fix a quaternion (qw ≥ 0)."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0.0:
        raise ValueError("zero quaternion")
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    if qw < 0.0 or (qw == 0.0 and (qx, qy, qz) < (0.0, 0.0, 0.0)):
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    return qw, qx, qy, qz


def _quat_to_matrix(qw: float, qx: float, qy: float, qz: float) -> tuple:
    """3×3 rotation matrix (row-major) from a normalized quaternion."""
    return (
        1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
        2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
        2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy),
    )


def _basis_from_normal_xref(nx, ny, nz, xref):
    """Orthonormal frame with +z = normal and +x = projected xref (auto when None)."""
    z = (nx, ny, nz)
    n = math.sqrt(sum(c * c for c in z))
    if n == 0.0:
        raise ValueError("zero plane normal")
    z = (z[0] / n, z[1] / n, z[2] / n)
    if xref is None:
        # auto: project world +X (or +Y when nearly parallel)
        xref = (1.0, 0.0, 0.0) if abs(z[0]) < 0.9 else (0.0, 1.0, 0.0)
    x = (xref[0], xref[1], xref[2])
    d = x[0] * z[0] + x[1] * z[1] + x[2] * z[2]
    x = (x[0] - d * z[0], x[1] - d * z[1], x[2] - d * z[2])
    nx = math.sqrt(sum(c * c for c in x))
    if nx < 1e-12:
        raise ValueError("x reference is parallel to the plane normal")
    x = (x[0] / nx, x[1] / nx, x[2] / nx)
    y = (z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0])
    return x, y, z


def _matrix_to_quaternion(m) -> tuple:
    """Quaternion from a 3×3 rotation matrix (robust branch on largest pivot)."""
    tr = m[0] + m[4] + m[8]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[7] - m[5]) / s
        qy = (m[2] - m[6]) / s
        qz = (m[3] - m[1]) / s
    elif m[0] > m[4] and m[0] > m[8]:
        s = math.sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0
        qw = (m[7] - m[5]) / s
        qx = 0.25 * s
        qy = (m[1] + m[3]) / s
        qz = (m[2] + m[6]) / s
    elif m[4] > m[8]:
        s = math.sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0
        qw = (m[2] - m[6]) / s
        qx = (m[1] + m[3]) / s
        qy = 0.25 * s
        qz = (m[5] + m[7]) / s
    else:
        s = math.sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0
        qw = (m[3] - m[1]) / s
        qx = (m[2] + m[6]) / s
        qy = (m[5] + m[7]) / s
        qz = 0.25 * s
    return _normalize_quaternion(qw, qx, qy, qz)


class CoordinateSystem:
    """A placed Cartesian frame in 3D model space.

    The XY plane is the designated planarity plane for cage tangents
    (z = plane normal; angles in XY polar around +z, from +x).
    Generic and reusable — cage semantics live in :class:`Corner`.
    """

    def __init__(
        self,
        position,
        orientation=None,
        *,
        id: Optional[str] = None,
    ) -> None:
        self.id = id if id is not None else f"cs-{contract_signature_seed()}"
        pos = tuple(float(c) for c in position)
        if len(pos) != 3:
            raise ValueError("position must have 3 components")
        self.position = pos
        if orientation is None:
            self.orientation = (1.0, 0.0, 0.0, 0.0)  # identity, canonical
        else:
            qw, qx, qy, qz = (float(c) for c in orientation)
            self.orientation = _normalize_quaternion(qw, qx, qy, qz)

    @classmethod
    def from_plane(
        cls, position, normal, x_ref=None, *, id: Optional[str] = None
    ) -> "CoordinateSystem":
        """Place a frame with +z = *normal* and +x = projected *x_ref* (auto when None)."""
        x, y, z = _basis_from_normal_xref(normal[0], normal[1], normal[2], x_ref)
        m = (x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2])
        # row-major matrix maps local axes onto these world columns:
        m = (x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2])
        quat = _matrix_to_quaternion(m)
        return cls(position, quat, id=id)

    def matrix(self) -> tuple:
        qw, qx, qy, qz = self.orientation
        return _quat_to_matrix(qw, qx, qy, qz)

    def apply_direction(self, v) -> tuple:
        m = self.matrix()
        return (
            m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
        )

    def apply_point(self, p) -> tuple:
        d = self.apply_direction(p)
        p0 = self.position
        return (d[0] + p0[0], d[1] + p0[1], d[2] + p0[2])

    def to_record(self) -> dict:
        rec = {
            "kind": "coordinate_system",
            "id": self.id,
            "position": {"x": self.position[0], "y": self.position[1], "z": self.position[2]},
            "orientation": {
                "qw": self.orientation[0], "qx": self.orientation[1],
                "qy": self.orientation[2], "qz": self.orientation[3],
            },
        }
        return rec

    def signature(self) -> str:
        return _contract.signature(self.to_record())


def contract_signature_seed() -> str:
    """Short id seed for auto-named frames (stable within a process)."""
    import uuid

    return uuid.uuid4().hex[:8]


class Corner:
    """Cage corner: binds a ``CoordinateSystem`` to curve terminals.

    ``angles`` maps ``curve_id`` to the tangent angle θ (radians) in the
    frame's XY polar coordinates — the associative record: an entry is
    the end-tangent direction for the curve departing this corner.

    v0.1 limitation: one terminal per curve per corner (loop curves with
    both ends on one corner need terminal-keyed records; documented
    extension).
    """

    def __init__(
        self, coordinate_system: CoordinateSystem, angles: dict, *, id: Optional[str] = None
    ) -> None:
        self.coordinate_system = coordinate_system
        self.angles = {str(k): float(v) for k, v in angles.items()}
        self.id = id if id is not None else f"corner-{contract_signature_seed()}"

    def tangent_for(self, curve_id) -> tuple:
        """End-tangent direction of *curve_id* at this corner, in 3D model space.

        Composition: the frame's XY polar direction at the recorded angle θ,
        mapped through the coordinate system's orientation. Unit length by
        construction.
        """
        key = str(curve_id)
        if key not in self.angles:
            raise KeyError(f"curve {curve_id!r} has no terminal at corner {self.id!r}")
        theta = self.angles[key]
        return self.coordinate_system.apply_direction((math.cos(theta), math.sin(theta), 0.0))

    def validate_terminals(self, min_spacing: float = 1e-6) -> None:
        """Raise when two terminals share (nearly) the same angle (degenerate cage)."""
        values = sorted(v % (2.0 * math.pi) for v in self.angles.values())
        for a, b in zip(values, values[1:] + [values[0] + 2.0 * math.pi]):
            if b - a < min_spacing:
                raise ValueError(
                    f"corner {self.id!r}: terminal angles degenerate (spacing {b - a:.3e})"
                )

    def to_record(self) -> dict:
        rec = {
            "kind": "corner",
            "id": self.id,
            "coordinate_system_id": self.coordinate_system.id,
            "angles": {str(k): v for k, v in self.angles.items()},
        }
        return rec

    def signature(self) -> str:
        return _contract.signature(self.to_record())
