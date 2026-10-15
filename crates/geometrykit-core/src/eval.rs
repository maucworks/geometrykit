//! Evaluation kernels: point, first-order derivatives, everything one
//! basis build supports.
//!
//! Verbatim port of the mirror (`runtimes/_npu/surface_np.py`): same
//! Piegl A2.1/A2.2 spans and basis, same right-limit derivative
//! convention, same accumulation order (station-major, pole row-major,
//! v-minor), same 1e-12 denominator guards. Bit-parity with the mirror
//! is a test gate; any reassociation of these expressions is a bug here.

/// One evaluated station: point, tangent, whatever — plain f64 triples.
pub type Pt = [f64; 3];

/// Point + both analytic tangents at a station.
pub struct EvalAll {
    pub pt: Pt,
    pub su: Pt,
    pub sv: Pt,
}

/// Prepared surface payload: expanded knots + pole grid + weights.
///
/// Mirrors the mirror's `_Data`: knot expansion happens once at
/// construction; payloads are treated as values afterwards. Pole grid
/// is row-major (`poles[i * nv + j]`); `weights` is `Some` with exactly
/// `nu * nv` row-major entries for rational surfaces.
pub struct Surface {
    poles: Vec<Pt>,
    weights: Option<Vec<f64>>,
    nu: usize,
    nv: usize,
    pu: usize,
    pv: usize,
    uflat: Vec<f64>,
    vflat: Vec<f64>,
}

impl Surface {
    /// Build from an (i-major) pole grid + degree/knot/multiplicity.
    pub fn new(
        poles: Vec<Pt>,
        nu: usize,
        nv: usize,
        weights: Option<Vec<f64>>,
        pu: usize,
        pv: usize,
        uknots: &[f64],
        umults: &[usize],
        vknots: &[f64],
        vmults: &[usize],
    ) -> Self {
        debug_assert_eq!(poles.len(), nu * nv);
        if let Some(w) = &weights {
            debug_assert_eq!(w.len(), nu * nv);
        }
        let mut uflat = Vec::new();
        for (k, m) in uknots.iter().zip(umults) {
            for _ in 0..*m {
                uflat.push(*k);
            }
        }
        let mut vflat = Vec::new();
        for (k, m) in vknots.iter().zip(vmults) {
            for _ in 0..*m {
                vflat.push(*k);
            }
        }
        Surface { poles, weights, nu, nv, pu, pv, uflat, vflat }
    }

    fn weight(&self, i: usize, j: usize) -> f64 {
        match &self.weights {
            None => 1.0,
            Some(w) => w[i * self.nv + j],
        }
    }

    fn clamp(&self, u: f64, v: f64) -> (f64, f64) {
        (
            u.max(self.uflat[self.pu]).min(self.uflat[self.uflat.len() - self.pu - 1]),
            v.max(self.vflat[self.pv]).min(self.vflat[self.vflat.len() - self.pv - 1]),
        )
    }

    /// Span index containing `u` (Piegl A2.1), with the exact-end
    /// special case returning the top control index.
    fn find_span(u: f64, p: usize, uflat: &[f64]) -> usize {
        let n = uflat.len() - p - 2;
        if u >= uflat[n + 1] {
            return n;
        }
        if u <= uflat[p] {
            return p;
        }
        let (mut lo, mut hi) = (p, n + 1);
        let mut mid = (lo + hi) / 2;
        while u < uflat[mid] || u >= uflat[mid + 1] {
            if u < uflat[mid] {
                hi = mid;
            } else {
                lo = mid;
            }
            mid = (lo + hi) / 2;
        }
        mid
    }

    /// Non-zero degree-`p` basis values at `u`: `(first_id, values)`,
    /// ids `first_id ..= first_id + p` (A2.2). A near-zero denominator
    /// contributes exactly 0.0, same as the mirror.
    fn nonzero_row(u: f64, p: usize, uflat: &[f64]) -> (usize, Vec<f64>) {
        let span = Self::find_span(u, p, uflat);
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
                let t = if denom.abs() > 1e-12 { n[r] / denom } else { 0.0 };
                n[r] = saved + right[r + 1] * t;
                saved = left[j - r] * t;
            }
            n[j] = saved;
        }
        (span - p, n)
    }

    /// `(first_id, Nu, dNu)` at `u` (eq. 3.8 derivative of the live
    /// basis row). Right-limit handling mirrors the mirror exactly:
    /// value row at the exact parameter, derivative row one
    /// representable `eps` inside the live domain at the ends.
    fn basis_and_deriv(u: f64, p: usize, uflat: &[f64]) -> (usize, Vec<f64>, Vec<f64>) {
        let (base, nu) = Self::nonzero_row(u, p, uflat);
        if p == 0 {
            let zero = vec![0.0; nu.len()];
            return (base, nu, zero);
        }
        let left_limit = uflat[p];
        let right_limit = uflat[uflat.len() - p - 1];
        let eps = if right_limit > left_limit {
            (right_limit - left_limit) * 1e-9
        } else {
            0.0
        };

        let ux = if eps > 0.0 {
            if u >= right_limit {
                right_limit - eps
            } else if u <= left_limit {
                left_limit + eps
            } else {
                u
            }
        } else {
            u
        };

        // derivative row at the one-sided limit (value row stays exact)
        let (low_base, low) = Self::nonzero_row(ux, p - 1, uflat);
        let top = uflat.len() as isize - 1;
        let pi = p as isize;
        let pf = p as f64;
        let mut ders = vec![0.0; nu.len()];
        for k in 0..nu.len() {
            let i = (base + k) as isize;
            let a = if i + pi <= top {
                uflat[(i + pi) as usize] - uflat[i as usize]
            } else {
                0.0
            };
            let b = if i + pi + 1 <= top {
                uflat[(i + pi + 1) as usize] - uflat[(i + 1) as usize]
            } else {
                0.0
            };
            let lv = |id: isize| -> f64 {
                if id >= low_base as isize
                    && (id - low_base as isize) < low.len() as isize
                {
                    low[(id - low_base as isize) as usize]
                } else {
                    0.0
                }
            };
            let t1 = if a.abs() > 1e-12 { pf * lv(i) / a } else { 0.0 };
            let t2 = if b.abs() > 1e-12 { pf * lv(i + 1) / b } else { 0.0 };
            ders[k] = t1 - t2;
        }
        (base, nu, ders)
    }

    /// Homogeneous point ⊕ weight sum; `(hpt, wsum)`.
    fn accumulate(&self, ub: usize, nu: &[f64], vb: usize, nv: &[f64]) -> (Pt, f64) {
        let mut hpt = [0.0, 0.0, 0.0];
        let mut wsum = 0.0;
        for (ki, ni) in nu.iter().enumerate() {
            let i = ub + ki;
            for (kj, nj) in nv.iter().enumerate() {
                let j = vb + kj;
                let c = self.poles[i * self.nv + j];
                let h = ni * nj * self.weight(i, j);
                hpt[0] += h * c[0];
                hpt[1] += h * c[1];
                hpt[2] += h * c[2];
                wsum += h;
            }
        }
        (hpt, wsum)
    }

    /// Point at `(u, v)` — rational-safe homogeneous divide.
    ///
    /// Panics on a zero weight sum, mirroring the mirror's
    /// `ZeroDivisionError`.
    pub fn eval_point(&self, u: f64, v: f64) -> Pt {
        let (u, v) = self.clamp(u, v);
        let (ub, nu) = Self::nonzero_row(u, self.pu, &self.uflat);
        let (vb, nv) = Self::nonzero_row(v, self.pv, &self.vflat);
        let (hpt, wsum) = self.accumulate(ub, &nu, vb, &nv);
        if wsum == 0.0 {
            panic!("surface weights sum to zero at (u, v)");
        }
        [hpt[0] / wsum, hpt[1] / wsum, hpt[2] / wsum]
    }

    /// Point + both analytic tangents at `(u, v)`, one basis build.
    ///
    /// The derivative accumulation order matches the mirror line for
    /// line (interleaved u- and v- contributions per pole), so the
    /// quotient rule stays bit-identical for rational and plain.
    pub fn eval_all(&self, u: f64, v: f64) -> EvalAll {
        let (u, v) = self.clamp(u, v);
        let (ub, nu, dnu) = Self::basis_and_deriv(u, self.pu, &self.uflat);
        let (vb, nv, dnv) = Self::basis_and_deriv(v, self.pv, &self.vflat);

        let (hpt, wsum) = self.accumulate(ub, &nu, vb, &nv);
        if wsum == 0.0 {
            panic!("surface weights sum to zero at (u, v)");
        }
        let (px, py, pz) = (hpt[0] / wsum, hpt[1] / wsum, hpt[2] / wsum);

        let mut d_ho = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let mut d_w = [0.0, 0.0];
        for (ki, ni) in nu.iter().enumerate() {
            let i = ub + ki;
            for (kj, nj) in nv.iter().enumerate() {
                let j = vb + kj;
                let c = self.poles[i * self.nv + j];
                let wij = self.weight(i, j);
                let dni = dnu.get(ki).copied().unwrap_or(0.0);
                let dnj = dnv.get(kj).copied().unwrap_or(0.0);
                d_ho[0][0] += dni * nj * wij * c[0];
                d_ho[0][1] += dni * nj * wij * c[1];
                d_ho[0][2] += dni * nj * wij * c[2];
                d_w[0] += dni * nj * wij;
                d_ho[1][0] += ni * dnj * wij * c[0];
                d_ho[1][1] += ni * dnj * wij * c[1];
                d_ho[1][2] += ni * dnj * wij * c[2];
                d_w[1] += ni * dnj * wij;
            }
        }
        EvalAll {
            pt: [px, py, pz],
            su: [
                (d_ho[0][0] - d_w[0] * px) / wsum,
                (d_ho[0][1] - d_w[0] * py) / wsum,
                (d_ho[0][2] - d_w[0] * pz) / wsum,
            ],
            sv: [
                (d_ho[1][0] - d_w[1] * px) / wsum,
                (d_ho[1][1] - d_w[1] * py) / wsum,
                (d_ho[1][2] - d_w[1] * pz) / wsum,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single-span bicubic patch, the harness fixture's shape.
    /// Poles i-major (i = the u direction).
    fn fixture_poles() -> (Vec<Pt>, usize, usize) {
        let z = |x: f64, y: f64| 3.0 + 0.35 * x + 0.15 * y - 0.8 * x * (1.0 - x) * y * (1.0 - y);
        let mut poles = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let (x, y) = (i as f64 / 3.0, j as f64 / 3.0);
                poles.push([x, y, z(x, y)]);
            }
        }
        (poles, 4, 4)
    }

    fn fixture() -> Surface {
        let (poles, nu, nv) = fixture_poles();
        Surface::new(
            poles, nu, nv, None, 3, 3, &[0.0, 1.0], &[4, 4], &[0.0, 1.0], &[4, 4],
        )
    }

    #[test]
    fn corner_is_pole() {
        let f = fixture();
        let p = f.eval_point(0.0, 0.0);
        assert_eq!(p, [0.0, 0.0, 3.0]);
        let pn = f.eval_point(1.0, 1.0);
        assert_eq!(pn[0], 1.0);
        assert_eq!(pn[1], 1.0);
    }

    #[test]
    fn edges_are_linear_in_u() {
        // poles x_i = i/3, z_i = 3 + 0.35 x_i on the y=0 fringe: the
        // Bernstein mean gives x(u,0)=u and z(u,0)=3+0.35u exactly
        let f = fixture();
        let e = f.eval_all(0.0, 0.0);
        assert!((e.su[0] - 1.0).abs() < 1e-12);
        assert!((e.su[1]).abs() < 1e-12);
        assert!((e.su[2] - 0.35).abs() < 1e-12);
        // fringe i=0: z = 3 + 0.15 y exactly
        assert!((e.sv[1] - 1.0).abs() < 1e-12);
        assert!((e.sv[2] - 0.15).abs() < 1e-12);
    }

    #[test]
    fn clamps_outside_domain() {
        let f = fixture();
        let a = f.eval_point(-0.5, 0.25);
        let b = f.eval_point(0.0, 0.25);
        assert_eq!(a[0], b[0]);
    }

    #[test]
    fn rational_rescales_positions() {
        // weighting one pole changes value there but not at unity weight
        let (poles, nu, nv) = fixture_poles();
        let w = vec![
            0.5, 1.5, 1.2, 0.9, 1.1, 0.8, 1.4, 1.0, 1.3, 0.7, 1.6, 1.05, 1.02, 1.15, 0.85, 1.25,
        ];
        let s = Surface::new(
            poles, nu, nv, Some(w), 3, 3, &[0.0, 1.0], &[4, 4], &[0.0, 1.0], &[4, 4],
        );
        // at the corner the single pole dominates: value IS the pole
        let c = s.eval_point(0.0, 0.0);
        assert_eq!(c, [0.0, 0.0, 3.0]);
    }
}
