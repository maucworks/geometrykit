//! `_rust` — the geometrykit accelerator extension module, built by
//! maturin from `crates/geometrykit-py`.
//!
//! Marshalling only: numpy/plain containers in, calls into
//! `geometrykit-core`, numbers out. No geometry logic lives here — the
//! kernels belong in `-core`, and anything wrong is wrong there.

use pyo3::exceptions::{PyRuntimeError, PyZeroDivisionError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

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

/// Read a Python `Surface`'s data into the core payload.
///
/// Marshalling only: attributes read, numbers converted, poles
/// flattened row-major in the mirror's (i = u-direction) outer order.
fn _payload(surface: &Bound<'_, PyAny>) -> PyResult<geometrykit_core::eval::Surface> {
    let cps = surface.getattr("control_points")?;
    let rows = cps.cast::<PyList>()?;
    let nu = rows.len();
    let mut poles = Vec::new();
    let mut nv: Option<usize> = None;
    for row in rows.iter() {
        let row = row.cast::<PyList>()?;
        let n = row.len();
        match nv {
            Some(0) => {}
            None => nv = Some(n),
            Some(m) if m != n => {
                return Err(PyValueError::new_err(
                    "Surface control_points is not rectangular",
                ))
            }
            _ => {}
        }
        for p in row.iter() {
            let x: f64 = p.getattr("x")?.extract()?;
            let y: f64 = p.getattr("y")?.extract()?;
            let z: f64 = p.getattr("z")?.extract()?;
            poles.push([x, y, z]);
        }
    }
    let nv = nv.unwrap_or(0);
    let pu: usize = surface.getattr("udegree")?.extract()?;
    let pv: usize = surface.getattr("vdegree")?.extract()?;
    let uknots: Vec<f64> = surface.getattr("uknots")?.extract()?;
    let umults: Vec<usize> = surface.getattr("umults")?.extract()?;
    let vknots: Vec<f64> = surface.getattr("vknots")?.extract()?;
    let vmults: Vec<usize> = surface.getattr("vmults")?.extract()?;
    let weights_raw = surface.getattr("_weights")?;
    let weights: Option<Vec<f64>> = if weights_raw.is_none() {
        None
    } else {
        Some(
            weights_raw
                .extract::<Vec<Vec<f64>>>()?
                .into_iter()
                .flatten()
                .collect(),
        )
    };
    Ok(geometrykit_core::eval::Surface::new(
        poles, nu, nv, weights, pu, pv, &uknots, &umults, &vknots, &vmults,
    ))
}

/// Evaluate the point grid for station pairs (`us[i], vs[i]`); flat
/// `[x, y, z] * n` out. One Python round-trip, whatever the grid size.
#[pyfunction]
fn eval_point_grid(
    surface: &Bound<'_, PyAny>,
    us: Vec<f64>,
    vs: Vec<f64>,
) -> PyResult<Vec<f64>> {
    if us.len() != vs.len() {
        return Err(PyValueError::new_err("us and vs must pair 1:1"));
    }
    let s = _payload(surface)?;
    let mut out = Vec::with_capacity(us.len() * 3);
    for (u, v) in us.iter().zip(&vs) {
        let p = s
            .eval_point(*u, *v)
            .map_err(PyZeroDivisionError::new_err)?;
        out.extend_from_slice(&p);
    }
    Ok(out)
}

/// Point + both analytic tangents per station pair; flat
/// `[x, y, z, sux, suy, suz, svx, svy, svz] * n` out.
#[pyfunction]
fn eval_all_grid(
    surface: &Bound<'_, PyAny>,
    us: Vec<f64>,
    vs: Vec<f64>,
) -> PyResult<Vec<f64>> {
    if us.len() != vs.len() {
        return Err(PyValueError::new_err("us and vs must pair 1:1"));
    }
    let s = _payload(surface)?;
    let mut out = Vec::with_capacity(us.len() * 9);
    for (u, v) in us.iter().zip(&vs) {
        let e = s
            .eval_all(*u, *v)
            .map_err(PyZeroDivisionError::new_err)?;
        out.extend_from_slice(&e.pt);
        out.extend_from_slice(&e.su);
        out.extend_from_slice(&e.sv);
    }
    Ok(out)
}

/// Nearest surface point per flat `[x, y, z] * n` target; flat
/// `[px, py, pz, u, v] * n` out. Default P0.1 knobs (m=n=6, iters=40).
#[pyfunction]
fn project_grid(surface: &Bound<'_, PyAny>, pts: Vec<f64>) -> PyResult<Vec<f64>> {
    if pts.len() % 3 != 0 {
        return Err(PyValueError::new_err("pts must be flat xyz triples"));
    }
    let s = _payload(surface)?;
    let mut out = Vec::with_capacity(pts.len() / 3 * 5);
    for c in pts.chunks_exact(3) {
        let (q, uv) = s
            .project_point(&[c[0], c[1], c[2]], 6, 6, 40)
            .map_err(PyZeroDivisionError::new_err)?;
        out.extend_from_slice(&q);
        out.push(uv.0);
        out.push(uv.1);
    }
    Ok(out)
}

/// Batched basis rows over an expanded knot vector: station-major
/// `(ids, values)` with the value stride `degree + 1`. Marshalling only.
#[pyfunction]
fn basis_row_grid(
    uflat: Vec<f64>,
    degree: usize,
    stations: Vec<f64>,
) -> PyResult<(Vec<usize>, Vec<f64>)> {
    let rows = geometrykit_core::basis::basis_rows(degree, &uflat, &stations);
    let stride = degree + 1;
    let mut ids = Vec::with_capacity(stations.len());
    let mut vals = Vec::with_capacity(stations.len() * stride);
    for (id, n) in rows {
        ids.push(id);
        vals.extend(n);
    }
    Ok((ids, vals))
}

/// Banded Cholesky over sparse rows, each row a flat `[j, v, j, v, ...]`
/// pair list; the factor comes back row-major `[[band row 0], [1], ...]`,
/// index-compatible with the mirror's list-of-lists. Marshalling only.
#[pyfunction]
fn banded_cholesky_grid(
    rows: Vec<Vec<f64>>,
    n: usize,
    bandwidth: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let parsed: Vec<Vec<(usize, f64)>> = rows
        .iter()
        .map(|row| {
            (0..row.len())
                .step_by(2)
                .map(|k| {
                    (
                        row[k].max(0.0) as usize,
                        row.get(k + 1).copied().unwrap_or(f64::NAN),
                    )
                })
                .collect::<Vec<(usize, f64)>>()
        })
        .collect();
    geometrykit_core::linalg::banded_cholesky(&parsed, n, bandwidth)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Banded solve with the factor from `banded_cholesky_grid`.
#[pyfunction]
fn banded_solve_grid(
    factor: Vec<Vec<f64>>,
    rhs: Vec<f64>,
    n: usize,
    bandwidth: usize,
) -> PyResult<Vec<f64>> {
    Ok(geometrykit_core::linalg::banded_solve(&factor, &rhs, n, bandwidth))
}

/// Global cubic interpolation: flat points `[x, y, z] * n`, optional
/// end tangents (also flat triples), optional explicit parameters.
/// Returns `(control points flat xyz, knots, multiplicities)`.
#[pyfunction]
fn interpolate_grid(
    pts: Vec<f64>,
    degree: usize,
    tan_start: Option<Vec<f64>>,
    tan_end: Option<Vec<f64>>,
    knots_param: Option<Vec<f64>>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    if pts.len() % 3 != 0 {
        return Err(PyValueError::new_err("pts must be flat xyz triples"));
    }
    let grid: Vec<[f64; 3]> = pts
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let ts = tan_start.map(|t| [t[0], t[1], t[2]]);
    let te = tan_end.map(|t| [t[0], t[1], t[2]]);
    let out = geometrykit_core::interp::interpolate(&grid, degree, ts, te, knots_param)
        .map_err(|e| match e {
            geometrykit_core::interp::InterpError::Arg(msg) => {
                PyValueError::new_err(msg)
            }
            geometrykit_core::interp::InterpError::Singular(msg) => {
                PyRuntimeError::new_err(msg)
            }
        })?;
    let flat = out
        .control_points
        .iter()
        .flat_map(|c| c.iter().copied())
        .collect::<Vec<f64>>();
    Ok((flat, out.knots, out.mults))
}

/// Section chains, flat: `[n_offsets, n_chains_off0, (n_pts, uv*2,
/// xyz*3)..., n_chains_off1, ...]`. Counts ride as exact f64 integers.
#[pyfunction]
fn section_levels_grid(
    surface: &Bound<'_, PyAny>,
    origin: Vec<f64>,
    normal: Vec<f64>,
    offsets: Vec<f64>,
    nu: usize,
    nv: usize,
) -> PyResult<Vec<f64>> {
    if origin.len() != 3 || normal.len() != 3 {
        return Err(PyValueError::new_err("origin/normal must be triples"));
    }
    let s = _payload(surface)?;
    let o = [origin[0], origin[1], origin[2]];
    let n = [normal[0], normal[1], normal[2]];
    let out = geometrykit_core::section::section_levels(&s, o, n, &offsets, nu, nv)
        .map_err(PyValueError::new_err)?;
    let mut flat = Vec::new();
    flat.push(out.len() as f64);
    for chains in &out {
        flat.push(chains.len() as f64);
        for chain in chains {
            flat.push(chain.uvs.len() as f64);
            for uv in &chain.uvs {
                flat.push(uv[0]);
                flat.push(uv[1]);
            }
            for p in &chain.pts {
                flat.extend_from_slice(p);
            }
        }
    }
    Ok(flat)
}

/// Python module bootstrap. Extension comes in as `geometrykit._rust`.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(bits_hex, m)?)?;
    m.add_function(wrap_pyfunction!(eval_point_grid, m)?)?;
    m.add_function(wrap_pyfunction!(eval_all_grid, m)?)?;
    m.add_function(wrap_pyfunction!(project_grid, m)?)?;
    m.add_function(wrap_pyfunction!(basis_row_grid, m)?)?;
    m.add_function(wrap_pyfunction!(banded_cholesky_grid, m)?)?;
    m.add_function(wrap_pyfunction!(banded_solve_grid, m)?)?;
    m.add_function(wrap_pyfunction!(interpolate_grid, m)?)?;
    m.add_function(wrap_pyfunction!(section_levels_grid, m)?)?;
    Ok(())
}
