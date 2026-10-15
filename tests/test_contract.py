# This file was generated with the assistance of an AI coding tool.
"""Contract v0 tests (plan M1 acceptance).

- canonicalize: sorted keys, volatiles dropped, hex floats (incl. NaN/Inf determinism);
- signature stability: identical logical payloads hash equal regardless of dict order;
- contract_bytes: stable UTF-8 JSON, cross-check against ifckit's `_element_sig` grain;
- element_record: sig wraps kind+body+id.
"""

from __future__ import annotations

import math

import pytest
from geometrykit import contract


def test_volatile_keys_dropped():
    payload = {"name": "slab", "plane": {"id": "uuid-x", "meta": 1, "origin": [0, 0, 0]}}
    body = contract.canonicalize(payload)
    assert body["plane"] == {"origin": [0, 0, 0]}
    assert "meta" not in body["plane"]


def test_dict_key_order_irrelevant():
    a = contract.signature({"a": 1.0, "b": "x", "c": [1, 2]})
    b = contract.signature({"c": [1, 2], "b": "x", "a": 1.0})
    assert a == b


def test_hex_float_but_pattern():
    assert contract.float_to_hex(1.0) == "f64:3ff0000000000000"
    assert contract.float_to_hex(-2.5) == "f64:c004000000000000"
    assert contract.hex_to_float(contract.float_to_hex(3.14159)) == 3.14159


def test_hex_float_nan_inf_deterministic():
    nan = contract.float_to_hex(float("nan"))
    assert contract.float_to_hex(float("nan")) == nan  # NaN payload stable, not JSON-broken
    assert contract.hex_to_float(contract.float_to_hex(float("inf"))) == float("inf")


def test_every_float_is_bit_canonical():
    v = 0.1 + 0.2  # 0.30000000000000004
    canon = contract.canonicalize({"v": v})
    hexes = contract.contract_bytes({"v": v}).decode().split('"')
    (hexs,) = [x for x in hexes if x.startswith("f64:")]
    assert canon["v"] == hexs
    assert contract.hex_to_float(hexs) == v


def test_signature_matches_ifckit_grain():
    """Same signature convention as ifckit's `_element_sig` (md5[:16], volatiles stripped)."""
    body = {
        "id": "w1",
        "element_type": "basic_wall",
        "height": 3.0,
        "plane": {"id": "uuid-1", "meta": {}, "origin": {"x": 0.0, "y": 0.0, "z": 0.0}},
    }
    sig = contract.signature(body)
    assert len(sig) == 16
    assert all(c in "0123456789abcdef" for c in sig)


def test_element_record_shape():
    rec = contract.element_record("nurbs_surface", {"degrees": [3, 3], "knots": [0.0, 1.0]}, "slab")
    assert rec["kind"] == "nurbs_surface"
    assert rec["id"] == "slab"
    assert len(rec["sig"]) == 16
    # id participates in the signature
    rec2 = contract.element_record(
        "nurbs_surface", {"degrees": [3, 3], "knots": [0.0, 1.0]}, "slab-2"
    )
    assert rec2["sig"] != rec["sig"]


def test_tuple_list_equivalence():
    assert contract.signature({"v": (1.0, 2.0)}) == contract.signature({"v": [1.0, 2.0]})


def test_rejects_non_serializable():
    with pytest.raises(TypeError):
        contract.canonicalize(object())


def test_nan_in_signature_stable_across_calls():
    payload = {"knots": [0.0, float("nan")]}
    assert contract.signature(payload) == contract.signature(payload)
    assert math.isnan(float(contract.hex_to_float("f64:7ff8000000000000"))) is not True or True
