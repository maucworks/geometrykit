# This file was generated with the assistance of an AI coding tool.
"""geometrykit contract v0 — canonical representation + signatures.

The kernel speaks one wire format across languages and engines: **plain
arrays + stable ids**, canonically serialized for hashing. Every engine
(`runtimes/_npu` numpy mirror, the `geometrykit._rust` accelerator) and the
planned `crates/geometrykit-core` must agree byte-for-byte on these
functions; cross-engine hash equality is a CI property test
(``tests/test_contract.py``).

Canonicalization rules (v0):

- dicts: keys sorted, all keys strings;
- sequences: lists (tuples/list equivalence);
- floats: replaced by ``"f64:<16 hex digits of the IEEE-754 bit pattern>"``
  — stable across Python repr, JSON round trips, and Rust ``f64::to_bits``
  (Python: ``struct.pack(">d", v).hex()``; Rust: ``format!("{:016x}", v.to_bits())``).
  NaN/Inf make their bit patterns deterministic instead of breaking JSON.
- volatile keys (``id``, ``meta`` on nested geometry objects) are dropped —
  ifckit's ``preview_session._VOLATILE_KEYS`` convention, generalized here;
- the canonical form is JSON (``sort_keys``), UTF-8 encoded; the signature
  is MD5 truncated to 16 hex chars (matches ifckit's ``_element_sig`` grain).
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any

#: Nested keys that carry no geometric meaning (per-run uuids, render meta).
VOLATILE_KEYS = frozenset({"id", "meta"})

__all__ = [
    "VOLATILE_KEYS",
    "canonicalize",
    "contract_bytes",
    "signature",
    "element_record",
    "float_to_hex",
    "hex_to_float",
]


def float_to_hex(v: float) -> str:
    """IEEE-754 bit pattern of *v* as ``f64:<16 hex digits>`` (deterministic for NaN/Inf)."""
    bits = struct.unpack(">Q", struct.pack(">d", float(v)))[0]
    return f"f64:{bits:016x}"


def hex_to_float(s: str) -> float:
    """Inverse of :func:`float_to_hex` (Rust ``f64::from_bits(u64::from_str_radix)``)."""
    if not s.startswith("f64:"):
        raise ValueError(f"not a canonical f64 payload: {s!r}")
    return struct.unpack(">d", struct.pack(">Q", int(s[4:], 16)))[0]


def canonicalize(value: Any) -> Any:
    """Return the canonical in-memory form: sorted string keys, hex floats, no volatiles."""
    if isinstance(value, dict):
        return {
            str(k): canonicalize(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            if k not in VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [canonicalize(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return float_to_hex(value)
        return float_to_hex(value)  # every float is bit-canonicalized
    if hasattr(value, "to_dict"):
        return canonicalize(value.to_dict())
    raise TypeError(f"cannot canonicalize {type(value).__name__}")


def contract_bytes(value: Any) -> bytes:
    """Exact canonical bytes of *value* (UTF-8 JSON, sorted keys, hex floats)."""

    text = json.dumps(canonicalize(value), sort_keys=True, separators=(",", ":"))
    return text.encode("utf-8")


def signature(value: Any) -> str:
    """MD5 of the canonical bytes, truncated to 16 hex chars (ifckit sig grain)."""
    return hashlib.md5(contract_bytes(value)).hexdigest()[:16]


def element_record(kind: str, body: Any, elem_id: str | None = None) -> dict:
    """Wrap geometry *body* in a contract record: ``{id, kind, body, sig}``.

    The signature covers kind + body + ``elem_id`` (the record's own ``id``
    key is excluded from hashing by the volatile-key rule, so the id rides
    under a non-volatile alias inside the signed payload).
    """
    record = {"kind": str(kind), "body": body, "id": elem_id}
    record["sig"] = signature(
        {"kind": record["kind"], "body": record["body"], "elem_id": record["id"]}
    )
    return record
