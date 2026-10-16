//! Adaptive kernel tessellation — OCC-free surface meshing.
//!
//! Verbatim port of the mirror (`runtimes/_npu/tessellate_np`): a
//! restricted quadtree over the parameter domain (dyadic lattice
//! stations, one basis build per eval), error-split on edge-midpoint
//! deviation vs `deflection` and corner-normal deviation vs
//! `angle_tol`, 2:1 balance, green closure fanning split edges from
//! the evaluated centroid. Bit-parity with the mirror is a test gate;
//! any reassociation of these expressions is a bug here.

use std::collections::{BTreeMap, HashMap};

use crate::eval::Surface;

/// Tessellated mesh: flat xyz vertices, flat unit normals, index
/// triples, flat uv tracks per vertex.
pub struct Mesh {
    pub verts: Vec<f64>,
    pub normals: Vec<f64>,
    pub tris: Vec<usize>,
    pub uvs: Vec<f64>,
}

fn station_uv(u0: f64, u1: f64, v0: f64, v1: f64, iu: i64, iv: i64, level: u32) -> (f64, f64) {
    let scale = 1.0 / ((1u64 << level) as f64);
    (
        u0 + (u1 - u0) * iu as f64 * scale,
        v0 + (v1 - v0) * iv as f64 * scale,
    )
}

type Station = (i64, i64, u32);
type Quad = (i64, i64, u32);

/// Prepared tessellation state: the surface, its domain, and the eval
/// cache (station -> (point, normal)).
struct Tess<'a> {
    surf: &'a Surface,
    u0: f64,
    u1: f64,
    v0: f64,
    v1: f64,
    deflection: f64,
    angle_tol: f64,
    max_depth: u32,
    cache: HashMap<Station, ([f64; 3], [f64; 3])>,
}

impl<'a> Tess<'a> {
    fn eval(&mut self, st: Station) -> ([f64; 3], [f64; 3]) {
        if let Some(hit) = self.cache.get(&st) {
            return *hit;
        }
        let (iu, iv, level) = st;
        let (u, v) = station_uv(self.u0, self.u1, self.v0, self.v1, iu, iv, level);
        let hit = match self.surf.eval_all(u, v) {
            Ok(e) => {
                let (su, sv) = (e.su, e.sv);
                let mut n = [
                    su[1] * sv[2] - su[2] * sv[1],
                    su[2] * sv[0] - su[0] * sv[2],
                    su[0] * sv[1] - su[1] * sv[0],
                ];
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if len > 0.0 {
                    n[0] /= len;
                    n[1] /= len;
                    n[2] /= len;
                }
                (e.pt, n)
            }
            Err(_) => ([f64::NAN; 3], [f64::NAN; 3]),
        };
        self.cache.insert(st, hit);
        hit
    }

    fn corners(i: i64, j: i64, level: u32) -> [Station; 4] {
        [
            (i, j, level),
            (i + 1, j, level),
            (i + 1, j + 1, level),
            (i, j + 1, level),
        ]
    }

    fn mid(a: Station, b: Station) -> Station {
        debug_assert_eq!(a.2, b.2);
        (a.0 + b.0, a.1 + b.1, a.2 + 1)
    }

    fn quad_error(&mut self, i: i64, j: i64, level: u32) -> (f64, f64) {
        let corners = Self::corners(i, j, level);
        let pts: Vec<[f64; 3]> = corners.iter().map(|c| self.eval(*c).0).collect();
        let norms: Vec<[f64; 3]> = corners.iter().map(|c| self.eval(*c).1).collect();
        let mut edge_dev: f64 = 0.0;
        for k in 0..4 {
            let a = corners[k];
            let b = corners[(k + 1) % 4];
            let mid = self.eval(Self::mid(a, b)).0;
            let (pa, pb) = (pts[k], pts[(k + 1) % 4]);
            edge_dev = edge_dev.max(
                (mid[0] - 0.5 * (pa[0] + pb[0])).abs()
                    + (mid[1] - 0.5 * (pa[1] + pb[1])).abs()
                    + (mid[2] - 0.5 * (pa[2] + pb[2])).abs(),
            );
        }
        // diagonals and centroid (fan spokes land here)
        let center = (corners[0].0 + corners[2].0, corners[0].1 + corners[2].1, corners[0].2 + 1);
        let cc = self.eval(center).0;
        for (pa, pb) in [(pts[0], pts[2]), (pts[1], pts[3])] {
            edge_dev = edge_dev.max(
                (cc[0] - 0.5 * (pa[0] + pb[0])).abs()
                    + (cc[1] - 0.5 * (pa[1] + pb[1])).abs()
                    + (cc[2] - 0.5 * (pa[2] + pb[2])).abs(),
            );
        }
        let bil = [
            (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4.0,
            (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4.0,
            (pts[0][2] + pts[1][2] + pts[2][2] + pts[3][2]) / 4.0,
        ];
        edge_dev = edge_dev.max(
            (cc[0] - bil[0]).abs() + (cc[1] - bil[1]).abs() + (cc[2] - bil[2]).abs(),
        );
        // fan spokes to all 8 boundary stations
        let mut boundary = Vec::with_capacity(8);
        boundary.extend_from_slice(&corners);
        for k in 0..4 {
            boundary.push(Self::mid(corners[k], corners[(k + 1) % 4]));
        }
        for (biu, biv, bl) in boundary {
            let lift: i64 = 1 << (center.2 - bl);
            let mstat = (center.0 + lift * biu, center.1 + lift * biv, center.2 + 1);
            let sm = self.eval(mstat).0;
            let bp = self.eval((biu, biv, bl)).0;
            edge_dev = edge_dev.max(
                (sm[0] - 0.5 * (cc[0] + bp[0])).abs()
                    + (sm[1] - 0.5 * (cc[1] + bp[1])).abs()
                    + (sm[2] - 0.5 * (cc[2] + bp[2])).abs(),
            );
        }
        // quad normal from the diagonals
        let d0 = [
            pts[2][0] - pts[0][0],
            pts[2][1] - pts[0][1],
            pts[2][2] - pts[0][2],
        ];
        let d1 = [
            pts[3][0] - pts[1][0],
            pts[3][1] - pts[1][1],
            pts[3][2] - pts[1][2],
        ];
        let qn = [
            d0[1] * d1[2] - d0[2] * d1[1],
            d0[2] * d1[0] - d0[0] * d1[2],
            d0[0] * d1[1] - d0[1] * d1[0],
        ];
        let ql = (qn[0] * qn[0] + qn[1] * qn[1] + qn[2] * qn[2]).sqrt();
        let mut normal_dev: f64 = 0.0;
        if ql > 0.0 {
            let qn = [qn[0] / ql, qn[1] / ql, qn[2] / ql];
            for n in &norms {
                let cos_a = (n[0] * qn[0] + n[1] * qn[1] + n[2] * qn[2]).clamp(-1.0, 1.0);
                normal_dev = normal_dev.max(cos_a.acos());
            }
        }
        (edge_dev, normal_dev)
    }
}

fn quads_touch(a: Quad, b: Quad) -> bool {
    let ((ai, aj, al), (bi, bj, bl)) = (a, b);
    let scale: i64 = 1 << al.max(bl);
    let (au0, au1) = (ai * (scale >> al), (ai + 1) * (scale >> al));
    let (av0, av1) = (aj * (scale >> al), (aj + 1) * (scale >> al));
    let (bu0, bu1) = (bi * (scale >> bl), (bi + 1) * (scale >> bl));
    let (bv0, bv1) = (bj * (scale >> bl), (bj + 1) * (scale >> bl));
    let share_v = (au1 == bu0 || bu1 == au0) && av0 < bv1 && bv0 < av1;
    let share_h = (av1 == bv0 || bv1 == av0) && au0 < bu1 && bu0 < au1;
    share_v || share_h
}

/// Triangulate *surf* adaptively; see the mirror for the method.
pub fn adaptive_mesh(
    surf: &Surface,
    deflection: f64,
    angle_tol: f64,
    max_depth: u32,
    base_level: u32,
) -> Result<Mesh, String> {
    if deflection <= 0.0 {
        return Err(format!("deflection must be positive, got {deflection}"));
    }
    if angle_tol <= 0.0 {
        return Err(format!("angle_tol must be positive, got {angle_tol}"));
    }
    if max_depth < base_level {
        return Err(format!("max_depth {max_depth} below base_level {base_level}"));
    }
    let (du0, du1) = surf.domain_u();
    let (dv0, dv1) = surf.domain_v();
    if !(du1 > du0 && dv1 > dv0) {
        return Err("degenerate parameter domain".to_string());
    }
    let mut tess = Tess {
        surf,
        u0: du0,
        u1: du1,
        v0: dv0,
        v1: dv1,
        deflection,
        angle_tol,
        max_depth,
        cache: HashMap::new(),
    };

    let n_base: i64 = 1 << base_level;
    let mut leaves: BTreeMap<Quad, ()> = BTreeMap::new();
    for i in 0..n_base {
        for j in 0..n_base {
            leaves.insert((i, j, base_level), ());
        }
    }

    // neighbour candidates at levels level-1..level+1
    let neighbors = |leaves: &BTreeMap<Quad, ()>, i: i64, j: i64, level: u32| -> Vec<Quad> {
        let mut found = Vec::new();
        let mut push = |key: Quad| {
            if leaves.contains_key(&key) && !found.contains(&key) {
                found.push(key);
            }
        };
        push((i - 1, j, level));
        push((i + 1, j, level));
        push((i, j - 1, level));
        push((i, j + 1, level));
        if level > 0 {
            push((i >> 1, j >> 1, level - 1));
        }
        push((2 * i - 1, 2 * j, level + 1));
        push((2 * i + 2, 2 * j, level + 1));
        push((2 * i - 1, 2 * j + 1, level + 1));
        push((2 * i + 2, 2 * j + 1, level + 1));
        push((2 * i, 2 * j - 1, level + 1));
        push((2 * i + 1, 2 * j - 1, level + 1));
        push((2 * i, 2 * j + 2, level + 1));
        push((2 * i + 1, 2 * j + 2, level + 1));
        found
    };

    loop {
        let mut marked = std::collections::BTreeSet::new();
        for &(i, j, level) in leaves.keys() {
            if level >= max_depth {
                continue;
            }
            let (edge_dev, normal_dev) = tess.quad_error(i, j, level);
            if edge_dev > deflection || normal_dev > angle_tol {
                marked.insert((i, j, level));
            }
        }
        if marked.is_empty() {
            break;
        }
        loop {
            let mut extra = std::collections::BTreeSet::new();
            for &(i, j, level) in leaves.keys() {
                if marked.contains(&(i, j, level)) {
                    continue;
                }
                for nb in neighbors(&leaves, i, j, level) {
                    if marked.contains(&nb) {
                        continue;
                    }
                    if nb.2 + 1 < level && nb.2 < max_depth {
                        if marked.iter().any(|&m| m.2 > nb.2 + 1 && quads_touch(nb, m)) {
                            extra.insert(nb);
                        }
                    }
                }
            }
            if extra.is_empty() {
                break;
            }
            marked.extend(extra);
        }
        for (i, j, level) in marked {
            leaves.remove(&(i, j, level));
            leaves.insert((2 * i, 2 * j, level + 1), ());
            leaves.insert((2 * i + 1, 2 * j, level + 1), ());
            leaves.insert((2 * i, 2 * j + 1, level + 1), ());
            leaves.insert((2 * i + 1, 2 * j + 1, level + 1), ());
        }
    }

    let mut verts: Vec<f64> = Vec::new();
    let mut normals: Vec<f64> = Vec::new();
    let mut uvs: Vec<f64> = Vec::new();
    let mut index_of: HashMap<Station, usize> = HashMap::new();
    let mut vertex = |tess: &mut Tess, st: Station| -> usize {
        if let Some(&idx) = index_of.get(&st) {
            return idx;
        }
        let (pp, nn) = tess.eval(st);
        let (px, py, pz) = (pp[0], pp[1], pp[2]);
        let (nx, ny, nz) = (nn[0], nn[1], nn[2]);
        let idx = verts.len() / 3;
        verts.extend_from_slice(&[px, py, pz]);
        normals.extend_from_slice(&[nx, ny, nz]);
        let (u, v) = station_uv(tess.u0, tess.u1, tess.v0, tess.v1, st.0, st.1, st.2);
        uvs.extend_from_slice(&[u, v]);
        index_of.insert(st, idx);
        idx
    };

    let covers = |candidate: Quad, a: Station, b: Station, level: u32| -> bool {
        let (ni, nj, _) = candidate;
        let ((aiu, aiv, _), (biu, biv, _)) = (a, b);
        if aiu == biu {
            let eu = 2 * aiu;
            let (lo, hi) = (2 * aiv.min(biv), 2 * aiv.max(biv));
            (ni == eu || ni + 1 == eu) && nj < hi && nj + 1 > lo
        } else {
            let ev = 2 * aiv;
            let (lo, hi) = (2 * aiu.min(biu), 2 * aiu.max(biu));
            (nj == ev || nj + 1 == ev) && ni < hi && ni + 1 > lo
        }
    };

    let mut tris: Vec<usize> = Vec::new();
    let keys: Vec<Quad> = leaves.keys().copied().collect();
    for (i, j, level) in keys {
        let corners = Tess::corners(i, j, level);
        let finer: Vec<Quad> = neighbors(&leaves, i, j, level)
            .into_iter()
            .filter(|nb| nb.2 == level + 1)
            .collect();
        let mut lp: Vec<Station> = vec![corners[0]];
        for k in 0..4 {
            let a = corners[k];
            let b = corners[(k + 1) % 4];
            if finer.iter().any(|nb| covers(*nb, a, b, level)) {
                lp.push(Tess::mid(a, b));
            }
            if k < 3 {
                lp.push(b);
            }
        }
        if lp.len() == 4 {
            let v: Vec<usize> = lp.iter().map(|p| vertex(&mut tess, *p)).collect();
            tris.extend_from_slice(&[v[0], v[1], v[2], v[0], v[2], v[3]]);
        } else {
            let center = (
                corners[0].0 + corners[2].0,
                corners[0].1 + corners[2].1,
                corners[0].2 + 1,
            );
            let c = vertex(&mut tess, center);
            let ring: Vec<usize> = lp.iter().map(|p| vertex(&mut tess, *p)).collect();
            for k in 0..ring.len() {
                tris.extend_from_slice(&[c, ring[k], ring[(k + 1) % ring.len()]]);
            }
        }
    }
    Ok(Mesh { verts, normals, tris, uvs })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Surface as EvalSurface;

    fn saddle() -> EvalSurface {
        let z = |x: f64, y: f64| {
            3.0 + 0.35 * x + 0.15 * y - 0.8 * x * (1.0 - x) * y * (1.0 - y)
        };
        let mut poles = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let (x, y) = (i as f64 / 3.0, j as f64 / 3.0);
                poles.push([x, y, z(x, y)]);
            }
        }
        EvalSurface::new(
            poles, 4, 4, None, 3, 3, &[0.0, 1.0], &[4, 4], &[0.0, 1.0], &[4, 4],
        )
    }

    fn edge_shares(tris: &[usize]) -> std::collections::HashMap<(usize, usize), usize> {
        let mut shares = std::collections::HashMap::new();
        for t in tris.chunks(3) {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let e = if a < b { (a, b) } else { (b, a) };
                *shares.entry(e).or_insert(0) += 1;
            }
        }
        shares
    }

    #[test]
    fn watertight_no_dangling() {
        let s = saddle();
        let m = adaptive_mesh(&s, 0.01, 0.5, 8, 2).expect("mesh");
        assert!(!m.tris.is_empty());
        let shares = edge_shares(&m.tris);
        assert!(shares.values().all(|&n| n <= 2));
        assert!(shares.values().any(|&n| n == 1)); // open patch boundary
        assert_eq!(m.uvs.len() / 2, m.verts.len() / 3);
    }

    #[test]
    fn deflection_bound_holds() {
        let s = saddle();
        let m = adaptive_mesh(&s, 0.001, 0.5, 8, 2).expect("mesh");
        // edge midpoints through the uv track, exact (no projection)
        let mut worst = 0.0f64;
        for t in m.tris.chunks(3) {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                let mu = 0.5 * (m.uvs[2 * a] + m.uvs[2 * b]);
                let mv = 0.5 * (m.uvs[2 * a + 1] + m.uvs[2 * b + 1]);
                // re-evaluate: reuse the surface directly
                let q = s.eval_point(mu, mv).expect("wsum");
                let mx = 0.5 * (m.verts[3 * a] + m.verts[3 * b]);
                let my = 0.5 * (m.verts[3 * a + 1] + m.verts[3 * b + 1]);
                let mz = 0.5 * (m.verts[3 * a + 2] + m.verts[3 * b + 2]);
                worst = worst.max(
                    (q[0] - mx).abs() + (q[1] - my).abs() + (q[2] - mz).abs(),
                );
            }
        }
        assert!(worst <= 0.001, "worst {worst}");
    }

    #[test]
    fn bad_arguments_raise() {
        let s = saddle();
        assert!(adaptive_mesh(&s, 0.0, 0.5, 8, 2).is_err());
        assert!(adaptive_mesh(&s, 0.01, 0.5, 1, 2).is_err());
    }
}
