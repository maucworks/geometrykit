//! Global curve interpolation — NURBS Book A9.1/A9.2 end conditions.
//!
//! Verbatim port of the mirror's `curve_np.interpolate` + its helpers
//! (`chord_params`, `averaged_inner_knots`, `_nonzero_basis`,
//! `_flat_unique`, `_gauss_solve`): same loop orders, same `_EPS`
//! guards, same full-Jordan elimination with partial pivoting, same
//! error messages. The mirror stays the reference; divergence here is
//! a test failure.

/// Mirror `_EPS`.
const EPS: f64 = 1e-12;

/// Interpolation error kind: argument-level (mirror raises
/// `ValueError`) vs system-level (mirror raises `RuntimeError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpError {
    /// Argument mismatch: raise ValueError.
    Arg(String),
    /// Singular system / lost row / layout: raise RuntimeError.
    Singular(String),
}

/// Cumulative chord-length params in `[0, 1]` (mirror `chord_params`:
/// plain L1 `abs(v - u)` on the Vec triple — the *taxicab* distance,
/// not Euclidean, by construction).
pub fn chord_params(points: &[[f64; 3]]) -> Vec<f64> {
    let mut total = 0.0f64;
    let mut acc = vec![0.0f64];
    for i in 1..points.len() {
        let u = points[i - 1];
        let v = points[i];
        // Vec.__abs__ == sqrt(x^2 + y^2 + z^2): Euclidean
        let d = ((v[0] - u[0]).powi(2) + (v[1] - u[1]).powi(2) + (v[2] - u[2]).powi(2)).sqrt();
        total += d;
        acc.push(total);
    }
    if total < EPS {
        return vec![0.0; points.len()];
    }
    acc.iter().map(|a| a / total).collect()
}

/// Averaged inner knots (mirror `averaged_inner_knots`, clamp tails).
pub fn averaged_inner_knots(params: &[f64], q: usize, count: usize, window_offset: isize) -> Vec<f64> {
    let mut out = Vec::with_capacity(count);
    for j in 1..=count {
        let mut s = 0.0;
        let mut used = 0usize;
        for step in 0..q {
            let a = j as isize + window_offset + step as isize;
            if 0 <= a && (a as usize) < params.len() {
                s += params[a as usize];
                used += 1;
            }
        }
        out.push(if used != 0 { s / q as f64 } else { params[params.len() - 1] });
    }
    out
}

/// Non-zero degree-p basis values at `u` -> `(first_id, values)`.
pub fn nonzero_basis(u: f64, p: usize, uflat: &[f64]) -> (usize, Vec<f64>) {
    let top = uflat.len() - p - 2;
    let span;
    if u >= uflat[top + 1] {
        span = top;
    } else if u <= uflat[p] {
        span = p;
    } else {
        let (mut lo, mut hi) = (p, top + 1);
        let mut mid = (lo + hi) / 2;
        while u < uflat[mid] || u >= uflat[mid + 1] {
            if u < uflat[mid] {
                hi = mid;
            } else {
                lo = mid;
            }
            mid = (lo + hi) / 2;
        }
        span = mid;
    }
    let mut left = vec![0.0; p + 1];
    let mut right = vec![0.0; p + 1];
    let mut n = vec![0.0; p + 1];
    n[0] = 1.0;
    for j in 1..=p {
        left[j] = u - uflat[span + 1 - j];
        right[j] = uflat[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            let q = if denom.abs() > EPS { n[r] / denom } else { 0.0 };
            n[r] = saved + right[r + 1] * q;
            saved = left[j - r] * q;
        }
        n[j] = saved;
    }
    (span - p, n)
}

/// Mirror `_flat_unique`.
fn flat_unique(flat: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let mut knots: Vec<f64> = Vec::new();
    let mut mults: Vec<usize> = Vec::new();
    for val in flat {
        if !knots.is_empty() && (val - knots[knots.len() - 1]).abs() < 1e-9 {
            let last = mults.len() - 1;
            mults[last] += 1;
        } else {
            knots.push(*val);
            mults.push(1);
        }
    }
    (knots, mults)
}

/// Dense Gaussian elimination with a 3-vector rhs (mirror
/// `_gauss_solve`): partial pivoting, full Jordan sweep.
///
/// Rows are `(coefs: Vec<(usize, f64)>, rhs: [f64; 3])`.
fn gauss_solve(rows: &[(Vec<(usize, f64)>, [f64; 3])], unknowns: usize) -> Result<Vec<[f64; 3]>, InterpError> {
    let n = rows.len();
    let dims = 3usize;
    let mut a = (0..n)
        .map(|_| vec![0.0f64; unknowns + dims])
        .collect::<Vec<_>>();
    for (r, (coefs, rhs)) in rows.iter().enumerate() {
        for (i, c) in coefs {
            if *i < unknowns {
                a[r][*i] = *c;
            }
        }
        for d in 0..dims {
            a[r][unknowns + d] = rhs[d];
        }
    }
    for col in 0..unknowns {
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in col + 1..n {
            let v = a[r][col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if a[piv][col].abs() < EPS {
            return Err(InterpError::Singular(
                "interpolation system is singular (degenerate data/tangents?)".to_string()
            ));
        }
        a.swap(col, piv);
        for r in 0..n {
            if r == col || a[r][col].abs() < EPS {
                continue;
            }
            let f = a[r][col] / a[col][col];
            for c in col..unknowns + dims {
                a[r][c] -= f * a[col][c];
            }
        }
    }
    let mut sol = vec![[0.0f64; 3]; unknowns];
    for col in 0..unknowns {
        let row = (0..n).find(|&r| a[r][col].abs() > EPS);
        let row = match row {
            Some(r) => r,
            None => return Err(InterpError::Singular("interpolation system lost a pivot row".to_string())),
        };
        for d in 0..dims {
            sol[col][d] = a[row][unknowns + d] / a[row][col];
        }
    }
    Ok(sol)
}

/// Interpolation result: control points + (knots, multiplicities).
pub struct Interpolated {
    pub control_points: Vec<[f64; 3]>,
    pub knots: Vec<f64>,
    pub mults: Vec<usize>,
}

/// Global interpolation through `points` (A9.1; the A9.2 tangent case
/// arrives through `tan_start/tan_end`, magnitude convention: natural
/// chord `(params[i+1]-params[i])/q`, unscaled).
///
/// Mirrors `curve_np.interpolate` including the clamped degree
/// (`min(degree, unknowns-1)`) and both error paths.
#[allow(clippy::too_many_arguments)]
pub fn interpolate(
    points: &[[f64; 3]],
    degree: usize,
    tan_start: Option<[f64; 3]>,
    tan_end: Option<[f64; 3]>,
    knots_param: Option<Vec<f64>>,
) -> Result<Interpolated, InterpError> {
    let r = points.len();
    if r < 2 {
        return Err(InterpError::Arg("Need at least 2 points".to_string()));
    }
    let with_t_start = tan_start.is_some();
    let with_t_end = tan_end.is_some();
    let extra = (with_t_start as usize) + (with_t_end as usize);
    let unknowns = r + extra;
    if unknowns < 2 {
        return Err(InterpError::Arg("Need at least 2 points".to_string()));
    }
    let q = 1.max(degree).min(unknowns - 1);
    let params = match knots_param {
        Some(k) => {
            if k.len() != r {
                return Err(InterpError::Arg("params length must match points".to_string()));
            }
            k
        }
        None => chord_params(points),
    };

    let inner = unknowns - q - 1;
    let wo: isize = if extra != 0 { -1 } else { 0 };
    let mut flat: Vec<f64> = Vec::with_capacity(unknowns + q + 1);
    flat.extend(std::iter::repeat(0.0).take(q + 1));
    flat.extend(averaged_inner_knots(&params, q, inner, wo));
    flat.extend(std::iter::repeat(1.0).take(q + 1));
    if flat.len() != unknowns + q + 1 {
        return Err(InterpError::Arg(format!(
            "interpolation knot layout broken: {} vs {}",
            flat.len(),
            unknowns + q + 1
        )));
    }

    let mut rows: Vec<(Vec<(usize, f64)>, [f64; 3])> = Vec::new();
    for k in 0..r {
        let (base, nu) = nonzero_basis(params[k], q, &flat);
        let coefs = (0..nu.len()).map(|j| (base + j, nu[j])).collect();
        rows.push((coefs, points[k]));
    }

    if with_t_start && r >= 2 {
        let den = params[1] - params[0];
        let sc = q as f64 / den;
        rows.insert(0, (vec![(0usize, -sc), (1usize, sc)], tan_start.unwrap()));
    }
    if with_t_end && r >= 2 {
        let den = params[r - 1] - params[r - 2];
        let sc = q as f64 / den;
        let last = unknowns - 2;
        rows.push((
            vec![(last, -sc), (last + 1, sc)],
            tan_end.unwrap(),
        ));
    }

    if rows.len() != unknowns {
        return Err(InterpError::Arg(format!(
            "rows {} vs unknowns {}: system shape mismatch",
            rows.len(),
            unknowns
        )));
    }
    let sol = gauss_solve(&rows, unknowns)?;
    let (knots, mults) = flat_unique(&flat);
    Ok(Interpolated {
        control_points: sol,
        knots,
        mults,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(sample: impl Fn(f64) -> [f64; 3], n: usize) -> Vec<[f64; 3]> {
        (0..n).map(|k| sample(k as f64 / (n - 1) as f64)).collect()
    }

    #[test]
    fn straight_line_is_linear() {
        let data = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let out = interpolate(&data, 3, None, None, None).expect("interp");
        // every pole on the line, endpoints exact
        assert_eq!(out.control_points[0], [0.0, 0.0, 0.0]);
        assert_eq!(out.control_points[3], [3.0, 0.0, 0.0]);
        for c in &out.control_points {
            assert!(c[1].abs() < 1e-9 && c[2].abs() < 1e-9);
        }
    }

    #[test]
    fn passes_through_all_points_degree1() {
        let data: Vec<[f64; 3]> = (0..5)
            .map(|k| [k as f64, (k as f64) * 0.5, 0.0])
            .collect();
        let out = interpolate(&data, 1, None, None, None).expect("interp");
        assert_eq!(out.control_points.len(), data.len());
        for (c, d) in out.control_points.iter().zip(&data) {
            assert_eq!(c, d);
        }
    }

    #[test]
    fn tangent_case_shapes_match() {
        let data = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0], [3.0, -1.0, 0.0]];
        let out = interpolate(&data, 3, Some([1.0, 0.0, 0.0]), Some([1.0, 0.0, 0.0]), None)
            .expect("interp");
        assert_eq!(out.control_points.len(), 6); // r + 2
        assert_eq!(out.knots.len(), 4); // 0, inner1, inner2, 1
    }

    #[test]
    fn singular_data_raises() {
        let data = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        // chord params collapse to zeros -> singular system
        assert!(interpolate(&data, 3, None, None, None).is_err());
    }

    #[test]
    fn interp_on_polygon_hits_vertices() {
        // zig-zag; degree clamps to 1 for 3 points? no, min(3, 2) = 2
        let data = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]];
        let out = interpolate(&data, 3, None, None, None).expect("interp");
        // endpoints exact
        assert_eq!(out.control_points[0], [0.0, 0.0, 0.0]);
        assert_eq!(out.control_points[2], [2.0, 0.0, 0.0]);
        let _ = pts(|t| [t, 0.0, 0.0], 3);
    }
}
