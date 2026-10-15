//! `_rust` — the geometrykit accelerator extension module, built by
//! maturin from `crates/geometrykit-py`.
//!
//! Marshalling only: numpy/plain floats in, calls into
//! `geometrykit-core`, numbers out. No geometry logic lives here — the
//! kernels belong in `-core`, and anything wrong is wrong there.

use pyo3::prelude::*;

/// Engine-version stamp of the core, so parity tests can hold the two
/// sides honest about what they were built against.
#[pyfunction]
fn core_version() -> String {
    format!("geometrykit-core {}", geometrykit_core::VERSION)
}

/// Bit-pattern of an f64 as the contract's canonical payload string,
/// delegated verbatim to core (no reimplementation here on purpose).
#[pyfunction]
fn bits_hex(x: f64) -> String {
    geometrykit_core::bits_hex(x)
}

/// Python module bootstrap. Extension comes in as `geometrykit._rust`.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(bits_hex, m)?)?;
    Ok(())
}
