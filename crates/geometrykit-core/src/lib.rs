//! geometrykit-core — the geometry kernels in pure Rust.
//!
//! Pure kernels over plain data: knot vectors up front, flat `f64`
//! arrays out, no allocation beyond what a return needs, no pyo3, no
//! IFC, no OCCT. The Python mirror in the `geometrykit` distribution is
//! the reference implementation; divergence between this crate and the
//! mirror is a test failure, never a feature. The contract layer
//! (hex-float bit patterns + stable hashing) is mirrored here so
//! cross-engine signatures can be compared bit-for-bit from both sides.
//!
//! Phase 1 carries no geometry yet — the parity layer only.

#![forbid(unsafe_code)]

/// Kernels crate version, kept in step with the distribution.
pub const VERSION: &str = "0.1.0";

/// IEEE-754 bit pattern of `x` in the contract's canonical form.
///
/// Byte-identical to the Python mirror's
/// `geometrykit.contract.float_to_hex`: `f64:` + 16 lowercase hex
/// digits of the big-endian bit pattern (NaN payloads carried as-is,
/// `-0.0` distinct from `0.0`).
#[must_use]
pub fn bits_hex(x: f64) -> String {
    format!("f64:{:016x}", x.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_hex_matches_python_nan() {
        assert_eq!(bits_hex(1.0), "f64:3ff0000000000000");
        assert_eq!(bits_hex(-2.5), "f64:c004000000000000");
    }

    #[test]
    fn signed_zero_distinct() {
        assert_ne!(bits_hex(0.0), bits_hex(-0.0));
    }

    #[test]
    fn subnormal_and_extremes_survive() {
        for v in [
            f64::MIN_POSITIVE,          // smallest normal
            f64::from_bits(1),          // smallest subnormal
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert_eq!(
                bits_hex(v),
                format!("f64:{:016x}", v.to_bits()),
                "roundtrip pattern for {v}"
            );
        }
    }
}
