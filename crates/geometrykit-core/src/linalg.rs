//! Banded SPD solvers: Cholesky factorisation + triangular solves.
//!
//! Verbatim port of the mirror's `linalg_np` (`_as_band`,
//! `banded_cholesky`, `banded_solve`): same upper-band storage
//! (`band[i][k] = A[i][i+k]`), same loop order, same `1e-14`
//! pivoting tolerance and error messages — the mirror stays the
//! reference and any reassociation here is a bug.

/// Band-assembly error: an entry sits beyond the declared bandwidth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BandError(pub String);

impl std::fmt::Display for BandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Rows as sparse dicts: `row_k = [(j, value), ...]` ascending.
type SparseRows<'a> = &'a [Vec<(usize, f64)>];

/// Upper-band storage: `band[i][k]` holds `A[i][i+k]` for
/// `k = 0..bandwidth`. Verbatim mirror `_as_band` semantics: only the
/// upper triangle is read (add both directions in the caller), and an
/// entry beyond the bandwidth is an error, never a silent drop.
pub fn as_band(
    rows: SparseRows<'_>,
    n: usize,
    bandwidth: usize,
) -> Result<Vec<Vec<f64>>, BandError> {
    let mut band = vec![vec![0.0; bandwidth + 1]; n];
    for (i, row) in rows.iter().enumerate() {
        for (j, value) in row {
            if *j < i {
                // the upper triangle is what the factorisation reads;
                // taking both would double every off-diagonal entry
                continue;
            }
            let offset = j - i;
            if offset > bandwidth {
                // dropping this silently would corrupt the matrix (it
                // happened: a two-step stencil on a reduced grid needs
                // 2*stride+2, not stride), so it is an error
                return Err(BandError(format!(
                    "entry ({i},{j}) is {offset} off the diagonal, \
                     beyond the bandwidth {bandwidth} — compute it from \
                     the rows"
                )));
            }
            band[i][offset] += value;
        }
    }
    Ok(band)
}

/// Cholesky factor (upper, bandwidth `b`) of a symmetric positive
/// definite banded matrix. Returns the band storage of `R`:
/// `r[i][k] = R[i][i+k]`.
pub fn banded_cholesky(
    rows: SparseRows<'_>,
    n: usize,
    bandwidth: usize,
) -> Result<Vec<Vec<f64>>, BandError> {
    let a = as_band(rows, n, bandwidth)?;
    let mut r = vec![vec![0.0; bandwidth + 1]; n];
    for j in 0..n {
        let lo = j.saturating_sub(bandwidth);
        let mut diagonal = a[j][0];
        for k in lo..j {
            diagonal -= r[k][j - k] * r[k][j - k];
        }
        if diagonal <= 1e-14 {
            return Err(BandError(format!(
                "matrix is not positive definite at row {j} (pivot \
                 {diagonal:.3e})"
            )));
        }
        r[j][0] = diagonal.sqrt();
        let top = (n - 1).min(j + bandwidth);
        for i in j + 1..=top {
            let mut total = a[j][i - j];
            for k in lo..j {
                if i - k <= bandwidth {
                    total -= r[k][j - k] * r[k][i - k];
                }
            }
            r[j][i - j] = total / r[j][0];
        }
    }
    Ok(r)
}

/// Solve `A x = rhs` given the factor from `banded_cholesky`.
pub fn banded_solve(
    factor: &[Vec<f64>],
    rhs: &[f64],
    n: usize,
    bandwidth: usize,
) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        // forward: R^T y = rhs
        let lo = i.saturating_sub(bandwidth);
        let mut total = rhs[i];
        for k in lo..i {
            total -= factor[k][i - k] * y[k];
        }
        y[i] = total / factor[i][0];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        // back: R x = y
        let top = (n - 1).min(i + bandwidth);
        let mut total = y[i];
        for k in i + 1..=top {
            total -= factor[i][k - i] * x[k];
        }
        x[i] = total / factor[i][0];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3-point Laplacian: tridiagonal(-1, 2, -1) SPD.
    fn tridiag(n: usize) -> Vec<Vec<(usize, f64)>> {
        (0..n)
            .map(|i| {
                let mut r = vec![(i, 2.0)];
                if i + 1 < n {
                    r.push((i + 1, -1.0));
                }
                r
            })
            .collect()
    }

    #[test]
    fn solves_the_3_point_laplacian() {
        let n = 6;
        let rows = tridiag(n);
        let f = banded_cholesky(&rows, n, 1).expect("spd");
        const RHS: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let x = banded_solve(&f, &RHS, n, 1);
        // A x == rhs within float noise
        for i in 0..n {
            let mut a_x = 2.0 * x[i];
            if i > 0 {
                a_x -= x[i - 1];
            }
            if i + 1 < n {
                a_x -= x[i + 1];
            }
            assert!((a_x - RHS[i]).abs() < 1e-10, "i={i} a_x={a_x}");
        }
    }

    #[test]
    fn non_spd_is_an_error() {
        let rows = vec![vec![(0usize, 0.0)]; 1];
        let err = banded_cholesky(&rows, 1, 1).expect_err("must fail");
        assert!(err.0.contains("not positive definite"), "{err}");
    }

    #[test]
    fn wide_entries_error_but_symmetry_dedupes() {
        // an upper entry beyond the bandwidth errors, never drops
        let rows = vec![vec![(0usize, 2.0), (9usize, 1.0)]];
        assert!(banded_cholesky(&rows, 3, 1).is_err());
        // a symmetric lower entry (row 1 col 0) is simply ignored:
        // the factorisation reads the upper triangle only
        let symmetric: Vec<Vec<(usize, f64)>> = (0..3)
            .map(|i| {
                let mut r = vec![(i, 2.0)];
                if i + 1 < 3 {
                    r.push((i + 1, -1.0));
                }
                if i > 0 {
                    r.push((i - 1, -1.0)); // mirrored: j < i, skipped
                }
                r
            })
            .collect();
        assert!(banded_cholesky(&symmetric, 3, 2).is_ok());
    }
}
