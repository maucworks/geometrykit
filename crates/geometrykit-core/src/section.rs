//! Plane × surface sections: marching in parameter space, Newton
//! polish, greedy chaining — the mirror's `section_np` ported verbatim.
//!
//! Section points are *evaluated* from the final tracks (exact
//! surface consistency); the only numerical error is the polished
//! plane residual (~1e-10). Same loop orders, same tolerances, same
//! saddle-cell pairing and greedy chaining semantics (pop-last
//! remaining), same bucket-free marching. Saddle-cell misrouting for
//! tangled topologies remains future work, as in the mirror.

use crate::eval::Surface;

/// Flagged None: python's `(x,)` marker.
type MaybeCrossing = Option<[f64; 2]>;

/// One chained contour: parameter track + evaluated 3D points.
pub struct Chain {
    pub pts: Vec<[f64; 3]>,
    pub uvs: Vec<[f64; 2]>,
}

/// Mirror `_polish` verbatim: iters=4 default, denominator/deci guards,
/// end-of-domain clamps.
fn polish(
    surf: &Surface,
    origin: &[f64; 3],
    normal: &[f64; 3],
    mut u: f64,
    mut v: f64,
) -> [f64; 2] {
    let iters = 4;
    for _ in 0..iters {
        let e = surf.eval_all(u, v);
        let (q, e_su, e_sv) = match e {
            Ok(e) => (e.pt, e.su, e.sv),
            Err(_) => break,
        };
        let f = (q[0] - origin[0]) * normal[0]
            + (q[1] - origin[1]) * normal[1]
            + (q[2] - origin[2]) * normal[2];
        let fu = e_su[0] * normal[0] + e_su[1] * normal[1] + e_su[2] * normal[2];
        let fv = e_sv[0] * normal[0] + e_sv[1] * normal[1] + e_sv[2] * normal[2];
        let denom = fu * fu + fv * fv;
        if denom < 1e-24 {
            break;
        }
        let step = f / denom;
        u -= step * fu;
        v -= step * fv;
        if f.abs() < 1e-10 {
            break;
        }
    }
    let uflat = surf.domain_u();
    let vflat = surf.domain_v();
    [u.min(uflat.1).max(uflat.0), v.min(vflat.1).max(vflat.0)]
}

fn signed(surf: &Surface, origin: &[f64; 3], normal: &[f64; 3], u: f64, v: f64) -> f64 {
    let q = surf.eval_point(u, v).expect("wsum for section field");
    (q[0] - origin[0]) * normal[0]
        + (q[1] - origin[1]) * normal[1]
        + (q[2] - origin[2]) * normal[2]
}

/// Mirror `_chain`: greedy endpoint chaining of segments into
/// polylines in the given order, exact-duplicate collapse, pop-last.
fn chain(
    segments: Vec<([f64; 2], [f64; 2])>,
    tol: f64,
) -> Vec<Vec<[f64; 2]>> {
    const EXACT: f64 = 1e-9;
    fn same(a: &[f64; 2], b: &[f64; 2]) -> bool {
        (a[0] - b[0]).abs() + (a[1] - b[1]).abs() <= EXACT
    }
    fn append_unique(chain: &mut Vec<[f64; 2]>, points: &[[f64; 2]]) {
        for point in points {
            if chain.is_empty() || !same(chain.last().unwrap(), point) {
                chain.push(*point);
            }
        }
    }
    let mut remaining: Vec<([f64; 2], [f64; 2])> = segments;
    let mut chains: Vec<Vec<[f64; 2]>> = Vec::new();
    while let Some(first) = remaining.pop() {
        let mut ch: Vec<[f64; 2]> = vec![first.0];
        append_unique(&mut ch, &[first.1]);
        let mut extended = true;
        while extended {
            extended = false;
            for end in 0..2 {
                let end_i = if end == 0 { 0usize } else { ch.len() - 1 };
                let anchor = ch[end_i];
                // best endpoint within tol among all remaining segments
                let mut best_e: Option<(usize, usize, f64)> = None;
                for (k, seg) in remaining.iter().enumerate() {
                    for e in 0..2 {
                        let cand = if e == 0 { seg.0 } else { seg.1 };
                        let dist =
                            (cand[0] - anchor[0]).abs() + (cand[1] - anchor[1]).abs();
                        if dist <= tol && best_e.map_or(true, |(_, _, bd)| dist < bd) {
                            best_e = Some((k, e, dist));
                        }
                    }
                }
                if let Some((k, e, _)) = best_e {
                    let seg = remaining.remove(k);
                    if end != 0 {
                        // mirror: `end == -1` appends, `end == 0`
                        // prepends — the token decides, not position
                        // (a degenerate first segment can make the
                        // chain length 1, where the two disagree)
                        let flipped = if e == 0 { [seg.0, seg.1] } else { [seg.1, seg.0] };
                        append_unique(&mut ch, &flipped);
                    } else {
                        let head = if e == 0 {
                            [seg.1, seg.0]
                        } else {
                            [seg.0, seg.1]
                        };
                        let mut new_head: Vec<[f64; 2]> = Vec::new();
                        append_unique(&mut new_head, &head);
                        let keep = ch.clone();
                        ch.clear();
                        for p in &new_head {
                            if !same(p, &keep[0]) {
                                ch.push(*p);
                            }
                        }
                        for p in &keep {
                            ch.push(*p);
                        }
                    }
                    extended = true;
                    break; // one consumption per round, as the mirror does
                }
            }
        }
        if ch.len() >= 2 {
            chains.push(ch);
        }
    }
    chains
}

/// Level sets of one scalar field: many parallel sections, one field
/// evaluation (mirror `section_levels` verbatim, empty offsets empty).
pub fn section_levels(
    surf: &Surface,
    plane_origin: [f64; 3],
    plane_normal: [f64; 3],
    offsets: &[f64],
    nu: usize,
    nv: usize,
) -> Result<Vec<Vec<Chain>>, String> {
    let length = (plane_normal[0] * plane_normal[0]
        + plane_normal[1] * plane_normal[1]
        + plane_normal[2] * plane_normal[2])
        .sqrt();
    if length < 1e-12 {
        return Err("section plane normal is degenerate".to_string());
    }
    let normal = [
        plane_normal[0] / length,
        plane_normal[1] / length,
        plane_normal[2] / length,
    ];
    let (u0, u1) = surf.domain_u();
    let (v0, v1) = surf.domain_v();

    // one scalar field for all levels
    let mut field: Vec<Vec<f64>> = Vec::with_capacity(nu + 1);
    for i in 0..=nu {
        let u = u0 + (u1 - u0) * i as f64 / nu as f64;
        let mut row = Vec::with_capacity(nv + 1);
        for j in 0..=nv {
            let v = v0 + (v1 - v0) * j as f64 / nv as f64;
            row.push(signed(surf, &plane_origin, &normal, u, v));
        }
        field.push(row);
    }

    let mut results = Vec::with_capacity(offsets.len());
    for offset in offsets {
        let shifted = [
            plane_origin[0] + offset * normal[0],
            plane_origin[1] + offset * normal[1],
            plane_origin[2] + offset * normal[2],
        ];
        let mut level: Vec<Vec<f64>> = Vec::with_capacity(field.len());
        for row in &field {
            let mut out_row = Vec::with_capacity(row.len());
            for value in row {
                let shifted_value = value - offset;
                out_row.push(if shifted_value != 0.0 { shifted_value } else { 1e-300 });
            }
            level.push(out_row);
        }

        let crossing = |i0: usize, j0: usize, i1: usize, j1: usize| -> [f64; 2] {
            let f0 = level[i0][j0];
            let f1 = level[i1][j1];
            let t = f0 / (f0 - f1);
            let u = u0 + (u1 - u0) * (i0 as f64 + t * (i1 as f64 - i0 as f64)) / nu as f64;
            let v = v0 + (v1 - v0) * (j0 as f64 + t * (j1 as f64 - j0 as f64)) / nv as f64;
            polish(surf, &shifted, &normal, u, v)
        };

        let mut segments: Vec<([f64; 2], [f64; 2])> = Vec::new();
        for i in 0..nu {
            for j in 0..nv {
                let c0 = (i, j);
                let c1 = (i + 1, j);
                let c2 = (i + 1, j + 1);
                let c3 = (i, j + 1);
                let mut crossings: Vec<MaybeCrossing> = Vec::new();
                for (a, b) in [(c0, c1), (c1, c2), (c2, c3), (c3, c0)] {
                    if (level[a.0][a.1] < 0.0) != (level[b.0][b.1] < 0.0) {
                        crossings.push(Some(crossing(a.0, a.1, b.0, b.1)));
                    }
                }
                if crossings.len() == 2 {
                    segments.push((crossings[0].unwrap(), crossings[1].unwrap()));
                } else if crossings.len() == 4 {
                    segments.push((crossings[0].unwrap(), crossings[1].unwrap()));
                    segments.push((crossings[2].unwrap(), crossings[3].unwrap()));
                }
            }
        }

        let tol = (u1 - u0) / nu as f64 + (v1 - v0) / nv as f64;
        let mut chains: Vec<Chain> = Vec::new();
        for uvs in chain(segments, tol) {
            let mut pts = Vec::with_capacity(uvs.len());
            for uv in &uvs {
                let q = surf.eval_point(uv[0], uv[1]).expect("wsum");
                pts.push(q);
            }
            chains.push(Chain { pts, uvs });
        }
        results.push(chains);
    }
    Ok(results)
}

/// Single-plane convenience: `offsets=[0]`.
pub fn section_plane(
    surf: &Surface,
    plane_origin: [f64; 3],
    plane_normal: [f64; 3],
    nu: usize,
    nv: usize,
) -> Result<Vec<Chain>, String> {
    Ok(section_levels(surf, plane_origin, plane_normal, &[0.0], nu, nv)?
        .remove(0))
}
