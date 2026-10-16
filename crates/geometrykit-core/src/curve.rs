//! One-dimensional B-spline curves: point evaluation and arc-length
//! division.
//!
//! Verbatim port of the mirror (`curve_np._nonzero_basis` for the
//! basis, `Curve.point_at`/`tangent_at` conventions, and
//! `curve_np.arclength_divide` for the table inversion): same Piegl
//! A2.2 basis, same normalized ``t ∈ [0, 1]`` mapping over the knot
//! domain, same chord table density, same binary search. Bit-parity
//! with the mirror is a test gate.

/// Rational-aware 1D B-spline curve.
pub struct Curve {
    poles: Vec<[f64; 3]>,
    weights: Option<Vec<f64>>,
    degree: usize,
    flat: Vec<f64>,
}

impl Curve {
    /// Build from control points + knot/multiplicity pairs.
    pub fn new(
        poles: Vec<[f64; 3]>,
        weights: Option<Vec<f64>>,
        degree: usize,
        knots: &[f64],
        mults: &[usize],
    ) -> Self {
        if let Some(w) = &weights {
            debug_assert_eq!(w.len(), poles.len());
        }
        let mut flat = Vec::new();
        for (k, m) in knots.iter().zip(mults) {
            for _ in 0..*m {
                flat.push(*k);
            }
        }
        Curve { poles, weights, degree, flat }
    }

    fn weight(&self, i: usize) -> f64 {
        match &self.weights {
            None => 1.0,
            Some(w) => w[i],
        }
    }

    fn domain(&self) -> (f64, f64) {
        (self.flat[self.degree], self.flat[self.flat.len() - self.degree - 1])
    }

    /// Point at normalized ``t ∈ [0, 1]`` (linear map over the knot
    /// domain, mirroring ``Curve._u_of_t``).
    pub fn eval_point(&self, t: f64) -> [f64; 3] {
        let (u0, u1) = self.domain();
        let u = u0 + t.max(0.0).min(1.0) * (u1 - u0);
        let (base, row) = crate::basis::basis_rows(self.degree, &self.flat, &[u])
            .remove(0);
        let mut hpt = [0.0, 0.0, 0.0];
        let mut wsum = 0.0;
        for (k, ni) in row.iter().enumerate() {
            let i = base + k;
            let h = ni * self.weight(i);
            hpt[0] += h * self.poles[i][0];
            hpt[1] += h * self.poles[i][1];
            hpt[2] += h * self.poles[i][2];
            wsum += h;
        }
        if wsum == 0.0 {
            return [f64::NAN; 3];
        }
        [hpt[0] / wsum, hpt[1] / wsum, hpt[2] / wsum]
    }
}

/// Arc-length division: normalized parameters in ``[0, 1]``.
///
/// Verbatim mirror of ``curve_np.arclength_divide``: dense chord
/// table, per-target binary search + linear interpolation. First
/// 0.0, last 1.0, ascending.
pub fn arclength_divide(
    curve: &Curve,
    total: f64,
    num: Option<usize>,
    dist: Option<f64>,
    even: bool,
) -> Result<Vec<f64>, String> {
    if total < 1e-12 {
        return Err("Cannot divide a zero-length curve".to_string());
    }
    let table_n: usize = (64).max(8 * num.unwrap_or(16));
    let mut cumulative = vec![0.0f64; table_n + 1];
    let mut prev = curve.eval_point(0.0);
    for i in 1..=table_n {
        let q = curve.eval_point(i as f64 / table_n as f64);
        cumulative[i] = cumulative[i - 1]
            + ((q[0] - prev[0]).powi(2)
            + (q[1] - prev[1]).powi(2)
            + (q[2] - prev[2]).powi(2))
            .sqrt();
        prev = q;
    }
    let table_total = cumulative[table_n];
    if table_total < 1e-12 {
        return Err("Cannot divide a zero-length curve".to_string());
    }
    let targets: Vec<f64> = match num {
        Some(n) => (0..n).map(|k| table_total * k as f64 / (n - 1) as f64).collect(),
        None => {
            let dist = dist.unwrap_or(table_total);
            if even {
                let segs = ((table_total / dist).ceil() as usize).max(1);
                (0..=segs).map(|k| table_total * k as f64 / segs as f64).collect()
            } else {
                let mut ts = Vec::new();
                let mut d = 0.0;
                while d < table_total {
                    ts.push(d);
                    d += dist;
                }
                ts.push(table_total);
                ts
            }
        }
    };
    let mut out = Vec::with_capacity(targets.len());
    for target in targets {
        let mut lo = 0usize;
        let mut hi = table_n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if cumulative[mid] < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo == 0 {
            out.push(0.0);
        } else if lo > table_n {
            out.push(1.0);
        } else {
            let span = cumulative[lo] - cumulative[lo - 1];
            if span <= 0.0 {
                out.push(lo as f64 / table_n as f64);
            } else {
                out.push(((lo - 1) as f64 + (target - cumulative[lo - 1]) / span)
                    / table_n as f64);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wiggle() -> Curve {
        // Hermite-like wiggle through chord-parametrized points is
        // overkill here; a plain 4-pole cubic suffices for the port
        let poles = vec![
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        Curve::new(poles, None, 3, &[0.0, 1.0], &[4, 4])
    }

    #[test]
    fn endpoints_exact_and_ascending() {
        let c = wiggle();
        let ts = arclength_divide(&c, 5.0, Some(9), None, true).expect("div");
        assert_eq!(ts.len(), 9);
        assert_eq!(ts[0], 0.0);
        assert_eq!(ts[8], 1.0);
        for w in ts.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn straight_line_uniform() {
        let c = Curve::new(
            vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            None,
            1,
            &[0.0, 1.0],
            &[2, 2],
        );
        let ts = arclength_divide(&c, 3.0, Some(4), None, true).expect("div");
        assert_eq!(ts, vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]);
    }

    #[test]
    fn zero_length_raises() {
        let c = wiggle();
        assert!(arclength_divide(&c, 0.0, Some(3), None, true).is_err());
    }
}
