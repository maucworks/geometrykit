//! Basis-row kernels: the row-assembly primitive the G1/G2 edge
//! constraint systems are built from.
//!
//! Verbatim port of the mirror's `curve_np._nonzero_basis` (Piegl A2.2
//! with the edge-aware span): same binary search, same loop order,
//! same `_EPS = 1e-12` denominators. The mirror stays the reference;
//! divergence here is a test failure.

/// Non-zero degree-`p` basis rows over an expanded knot vector —
/// one `(first_id, values)` per station, ids `first_id ..= first_id + p`
/// (A2.2, Piegl's `u == U[n+1] -> n` end case included).
pub fn basis_rows(p: usize, uflat: &[f64], stations: &[f64]) -> Vec<(usize, Vec<f64>)> {
    stations
        .iter()
        .map(|&t| {
            let span = find_span(t, p, uflat);
            let mut left = vec![0.0; p + 1];
            let mut right = vec![0.0; p + 1];
            let mut n = vec![0.0; p + 1];
            n[0] = 1.0;
            for j in 1..=p {
                left[j] = t - uflat[span + 1 - j];
                right[j] = uflat[span + j] - t;
                let mut saved = 0.0;
                for r in 0..j {
                    let denom = right[r + 1] + left[j - r];
                    let q = if denom.abs() > 1e-12 { n[r] / denom } else { 0.0 };
                    n[r] = saved + right[r + 1] * q;
                    saved = left[j - r] * q;
                }
                n[j] = saved;
            }
            (span - p, n)
        })
        .collect()
}

/// Span index containing `u`, Piegl A2.1 edge-aware variant (the
/// top-control special case on the exact right limit).
fn find_span(u: f64, p: usize, uflat: &[f64]) -> usize {
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
    span
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_of_unity_everywhere() {
        // multi-span cubic: interior knots at .25/.5/.75
        let uflat: Vec<f64> = [
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.25, 0.25],
            vec![0.5, 0.5],
            vec![0.75, 0.75],
            vec![1.0, 1.0, 1.0, 1.0],
        ]
        .concat();
        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let rows = basis_rows(3, &uflat, &[t]);
            let (_base, n) = &rows[0];
            let s: f64 = n.iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "t={t} sum={s}");
        }
    }

    #[test]
    fn exact_end_hits_last_live_span() {
        let uflat: Vec<f64> = [vec![0.0; 4], vec![1.0; 4]].concat();
        let (base, n) = &basis_rows(3, &uflat, &[1.0])[0];
        assert_eq!(*base, 0); // the live row is the last p+1 controls
        assert!((n[3] - 1.0).abs() < 1e-10);
        assert!((n[0]).abs() < 1e-10);
    }
}
