//! Binned-SAH bounding volume hierarchies in the flat layout
//! `shaders/bvh.wgsl` traverses, two levels of them: a [`Bvh`] over each
//! mesh's triangles in its own space, built when the mesh is made, and a
//! [`Tlas`] over the instances placing the meshes, built when one moves.
//!
//! Built on the CPU (never per frame) and uploaded as storage buffers. Nodes are depth-first with siblings side by side:
//! an interior node's children are `a` and `a + 1`; a leaf holds `count`
//! triangles from `a`. The tests trace it on the CPU as `bvh.wgsl` does.

use super::vector::V3;
use bytemuck::{Pod, Zeroable};

/// Bins a split candidate is chosen among, per axis.
const BINS: usize = 16;
/// Leaves never hold more than this unless the depth limit forces it.
const MAX_LEAF: usize = 8;
/// The deepest node: the GPU traversal's stack holds `MAX_DEPTH + 1`
/// entries (`STACK` in bvh.wgsl).
pub const MAX_DEPTH: u32 = 60;

/// A node: its bounds, and either its first child (count 0) or its first
/// triangle (count > 0).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Node {
    /// Bounds' minimum corner.
    pub min: [f32; 3],
    /// First child (interior) or first triangle (leaf).
    pub a: u32,
    /// Bounds' maximum corner.
    pub max: [f32; 3],
    /// Triangles in a leaf; 0 for an interior node.
    pub count: u32,
}

/// A triangle as the GPU tests it: a corner and the two edges from it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Triangle {
    /// The first corner (w unused).
    pub v0: [f32; 4],
    /// The second corner minus the first.
    pub e1: [f32; 4],
    /// The third corner minus the first.
    pub e2: [f32; 4],
}

/// The hierarchy.
#[derive(Clone, Debug, PartialEq)]
pub struct Bvh {
    /// The nodes, root first. Never empty.
    pub nodes: Vec<Node>,
    /// The triangles in leaf order. Never empty: an empty scene is one leaf
    /// holding one degenerate triangle, which nothing hits.
    pub triangles: Vec<Triangle>,
    /// Each stored triangle's index in the input.
    pub order: Vec<u32>,
}

#[derive(Clone, Copy)]
struct Bounds {
    min: V3,
    max: V3,
}

impl Bounds {
    const EMPTY: Bounds = Bounds {
        min: V3::splat(f32::INFINITY),
        max: V3::splat(f32::NEG_INFINITY),
    };

    fn grow(&mut self, p: V3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    fn join(&mut self, b: &Bounds) {
        self.min = self.min.min(b.min);
        self.max = self.max.max(b.max);
    }

    /// Half the surface area (the SAH's relative cost); 0 when empty.
    fn half_area(&self) -> f32 {
        let d = self.max - self.min;
        if d.x < 0.0 {
            return 0.0;
        }
        d.x * d.y + d.y * d.z + d.z * d.x
    }
}

impl Bvh {
    /// The hierarchy over `triangles` (corners in the space the rays it
    /// traces come in: a mesh's own).
    pub fn build(triangles: &[[[f32; 3]; 3]]) -> Bvh {
        if triangles.is_empty() {
            return Bvh {
                nodes: vec![Node {
                    count: 1,
                    ..Node::default()
                }],
                triangles: vec![Triangle::default()],
                order: Vec::new(),
            };
        }
        let corners: Vec<[V3; 3]> = triangles.iter().map(|t| t.map(V3::of)).collect();
        let boxes: Vec<Bounds> = corners
            .iter()
            .map(|c| {
                let mut b = Bounds::EMPTY;
                c.iter().for_each(|&p| b.grow(p));
                b
            })
            .collect();
        let (nodes, order) = tree(&boxes);
        let triangles = order
            .iter()
            .map(|&i| {
                let [a, b, c] = corners[i as usize];
                Triangle {
                    v0: a.with_w(0.0),
                    e1: (b - a).with_w(0.0),
                    e2: (c - a).with_w(0.0),
                }
            })
            .collect();
        Bvh {
            nodes,
            triangles,
            order,
        }
    }

    /// Its bounds (min, max); a point at the origin when empty.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        (self.nodes[0].min, self.nodes[0].max)
    }
}

/// The top level over the instances of meshes: nodes as [`Bvh`]'s, a leaf
/// holding `count` instances from `a` in [`Tlas::order`].
#[derive(Clone, Debug, PartialEq)]
pub struct Tlas {
    /// The nodes, root first. Never empty: with no instances, one leaf of
    /// empty bounds, which no ray enters.
    pub nodes: Vec<Node>,
    /// Each leaf entry's instance.
    pub order: Vec<u32>,
}

impl Tlas {
    /// The top level over the instances' world bounds `boxes` (min, max).
    pub fn build(boxes: &[([f32; 3], [f32; 3])]) -> Tlas {
        if boxes.is_empty() {
            let empty = Node {
                min: [f32::INFINITY; 3],
                max: [f32::NEG_INFINITY; 3],
                a: 0,
                count: 1,
            };
            return Tlas {
                nodes: vec![empty],
                order: Vec::new(),
            };
        }
        let boxes: Vec<Bounds> = (boxes.iter())
            .map(|&(min, max)| Bounds {
                min: V3::of(min),
                max: V3::of(max),
            })
            .collect();
        let (nodes, order) = tree(&boxes);
        Tlas { nodes, order }
    }

    /// The instances' bounds (min, max); none without instances.
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        (!self.order.is_empty()).then(|| (self.nodes[0].min, self.nodes[0].max))
    }
}

/// The nodes over `boxes` (not empty), depth-first with siblings side by
/// side, and each leaf entry's index in `boxes`.
fn tree(boxes: &[Bounds]) -> (Vec<Node>, Vec<u32>) {
    let centroids: Vec<V3> = boxes.iter().map(|b| (b.min + b.max) * 0.5).collect();
    let mut index: Vec<u32> = (0..boxes.len() as u32).collect();
    let mut nodes = vec![Node::default()];
    // (node, start, end, depth) still to split.
    let mut todo = vec![(0usize, 0usize, index.len(), 0u32)];
    while let Some((node, start, end, depth)) = todo.pop() {
        let mut bounds = Bounds::EMPTY;
        let mut centre = Bounds::EMPTY;
        for &i in &index[start..end] {
            bounds.join(&boxes[i as usize]);
            centre.grow(centroids[i as usize]);
        }
        nodes[node].min = bounds.min.to_array();
        nodes[node].max = bounds.max.to_array();
        let count = end - start;
        let split = if count <= 2 || depth >= MAX_DEPTH {
            None
        } else {
            split(&mut index[start..end], boxes, &centroids, &centre, &bounds)
        };
        match split {
            Some(mid) => {
                let left = nodes.len();
                nodes.push(Node::default());
                nodes.push(Node::default());
                nodes[node].a = left as u32;
                nodes[node].count = 0;
                todo.push((left + 1, start + mid, end, depth + 1));
                todo.push((left, start, start + mid, depth + 1));
            }
            None => {
                nodes[node].a = start as u32;
                nodes[node].count = count as u32;
            }
        }
    }
    (nodes, index)
}

/// Partitions `index` by the cheapest binned SAH split and returns the
/// split point; none when a leaf is cheaper (and small enough).
fn split(
    index: &mut [u32],
    boxes: &[Bounds],
    centroids: &[V3],
    centre: &Bounds,
    bounds: &Bounds,
) -> Option<usize> {
    let count = index.len();
    let extent = centre.max - centre.min;
    let mut best: Option<(f32, usize, usize)> = None; // (cost, axis, first right bin)
    for axis in 0..3 {
        if extent[axis] <= 0.0 {
            continue;
        }
        let scale = BINS as f32 / extent[axis];
        let bin_of = |c: V3| (((c[axis] - centre.min[axis]) * scale) as usize).min(BINS - 1);
        let mut bins = [(Bounds::EMPTY, 0usize); BINS];
        for &i in index.iter() {
            let b = &mut bins[bin_of(centroids[i as usize])];
            b.0.join(&boxes[i as usize]);
            b.1 += 1;
        }
        // Sweep from the right, then evaluate each cut from the left.
        let mut right_area = [0.0f32; BINS];
        let mut right_count = [0usize; BINS];
        let mut acc = Bounds::EMPTY;
        let mut n = 0;
        for k in (1..BINS).rev() {
            acc.join(&bins[k].0);
            n += bins[k].1;
            right_area[k] = acc.half_area();
            right_count[k] = n;
        }
        let mut acc = Bounds::EMPTY;
        let mut n = 0;
        for k in 0..BINS - 1 {
            acc.join(&bins[k].0);
            n += bins[k].1;
            let (rn, ra) = (right_count[k + 1], right_area[k + 1]);
            if n == 0 || rn == 0 {
                continue;
            }
            let cost = acc.half_area() * n as f32 + ra * rn as f32;
            if best.is_none_or(|b| cost < b.0) {
                best = Some((cost, axis, k + 1));
            }
        }
    }
    let leaf_cost = bounds.half_area() * count as f32;
    match best {
        Some((cost, axis, cut)) if cost < leaf_cost || count > MAX_LEAF => {
            let scale = BINS as f32 / extent[axis];
            let bin_of = |c: V3| (((c[axis] - centre.min[axis]) * scale) as usize).min(BINS - 1);
            let mid = partition(index, |i| bin_of(centroids[i as usize]) < cut);
            (mid > 0 && mid < count).then_some(mid)
        }
        Some(_) => None,
        // Every centroid in one place: halve by order if too many.
        None if count > MAX_LEAF => Some(count / 2),
        None => None,
    }
}

/// Moves the items `left` accepts to the front; returns how many.
fn partition(items: &mut [u32], left: impl Fn(u32) -> bool) -> usize {
    let mut mid = 0;
    for k in 0..items.len() {
        if left(items[k]) {
            items.swap(k, mid);
            mid += 1;
        }
    }
    mid
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Below this a ray's transmittance counts as nothing (`OPAQUE` in
    /// bvh.wgsl).
    const OPAQUE: f32 = 1e-3;

    /// A ray's closest hit.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Hit {
        /// The distance along the ray, in units of its direction.
        t: f32,
        /// The triangle's index in the input.
        triangle: u32,
    }

    /// The traversals `bvh.wgsl` makes, on the CPU.
    impl Bvh {
        /// The closest hit along `origin + t·dir` with `0 < t < t_max`.
        fn intersect(&self, origin: [f32; 3], dir: [f32; 3], t_max: f32) -> Option<Hit> {
            let mut best: Option<Hit> = None;
            let mut limit = t_max;
            self.traverse(origin, dir, |stored, t_hit| {
                if t_hit < limit {
                    limit = t_hit;
                    best = Some(Hit {
                        t: t_hit,
                        triangle: self.order[stored as usize],
                    });
                }
                (false, limit)
            });
            best
        }

        /// Whether anything lies along `origin + t·dir` with `0 < t < t_max`.
        fn occluded(&self, origin: [f32; 3], dir: [f32; 3], t_max: f32) -> bool {
            let mut hit = false;
            self.traverse(origin, dir, |_, t| {
                if t < t_max {
                    hit = true;
                }
                (hit, t_max)
            });
            hit
        }

        /// What gets through along `origin + t·dir` with `0 < t < t_max`: the
        /// product of `through(triangle)` (its index in the input) over the
        /// triangles there, 0 once below [`OPAQUE`] (`transmittance` in
        /// bvh.wgsl).
        fn transmittance(
            &self,
            origin: [f32; 3],
            dir: [f32; 3],
            t_max: f32,
            through: impl Fn(u32) -> [f32; 3],
        ) -> [f32; 3] {
            let mut kept = [1.0f32; 3];
            self.traverse(origin, dir, |stored, t| {
                if t < t_max {
                    let pass = through(self.order[stored as usize]);
                    kept = std::array::from_fn(|i| kept[i] * pass[i]);
                    if kept.iter().all(|&k| k < OPAQUE) {
                        kept = [0.0; 3];
                        return (true, t_max);
                    }
                }
                (false, t_max)
            });
            kept
        }

        /// Visits the triangles a ray hits; `found(stored index, t)` returns
        /// whether to stop and the distance boxes are still tested to.
        fn traverse(
            &self,
            origin: [f32; 3],
            dir: [f32; 3],
            mut found: impl FnMut(u32, f32) -> (bool, f32),
        ) {
            let o = V3::of(origin);
            let d = V3::of(dir);
            let inv = V3::ONE / d;
            let mut limit = f32::INFINITY;
            let mut stack = vec![0u32];
            while let Some(n) = stack.pop() {
                let node = &self.nodes[n as usize];
                if !hits_box(node, o, inv, limit) {
                    continue;
                }
                if node.count == 0 {
                    stack.push(node.a);
                    stack.push(node.a + 1);
                    continue;
                }
                for k in node.a..node.a + node.count {
                    if let Some(t) = hit_triangle(&self.triangles[k as usize], o, d) {
                        let (stop, l) = found(k, t);
                        if stop {
                            return;
                        }
                        limit = l;
                    }
                }
            }
        }
    }

    /// The slab test against a node's bounds, up to `limit`; an empty box
    /// (min above max) is entered by no ray.
    fn hits_box(node: &Node, o: V3, inv: V3, limit: f32) -> bool {
        let t0 = (V3::of(node.min) - o) * inv;
        let t1 = (V3::of(node.max) - o) * inv;
        let near = t0.min(t1).max_element().max(0.0);
        let far = t0.max(t1).min_element().min(limit);
        near <= far && (0..3).all(|k| node.min[k] <= node.max[k])
    }

    /// Möller–Trumbore: the hit distance, when positive.
    fn hit_triangle(tri: &Triangle, o: V3, d: V3) -> Option<f32> {
        let v0 = V3::new(tri.v0[0], tri.v0[1], tri.v0[2]);
        let e1 = V3::new(tri.e1[0], tri.e1[1], tri.e1[2]);
        let e2 = V3::new(tri.e2[0], tri.e2[1], tri.e2[2]);
        let p = d.cross(e2);
        let det = e1.dot(p);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv = 1.0 / det;
        let s = o - v0;
        let u = s.dot(p) * inv;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = s.cross(e1);
        let v = d.dot(q) * inv;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = e2.dot(q) * inv;
        (t > 0.0).then_some(t)
    }

    /// A small deterministic generator (xorshift), uniform in [0, 1).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }

        fn point(&mut self, scale: f32) -> [f32; 3] {
            [0, 1, 2].map(|_| (self.next() * 2.0 - 1.0) * scale)
        }
    }

    fn soup(rng: &mut Rng, n: usize) -> Vec<[[f32; 3]; 3]> {
        (0..n)
            .map(|_| {
                let c = rng.point(5.0);
                let corner = |r: &mut Rng| {
                    let o = r.point(0.6);
                    [c[0] + o[0], c[1] + o[1], c[2] + o[2]]
                };
                [corner(rng), corner(rng), corner(rng)]
            })
            .collect()
    }

    fn brute(tris: &[[[f32; 3]; 3]], o: [f32; 3], d: [f32; 3], t_max: f32) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        for (i, t) in tris.iter().enumerate() {
            let a = V3::of(t[0]);
            let edge = |b: [f32; 3]| {
                let e = V3::of(b) - a;
                [e.x, e.y, e.z, 0.0]
            };
            let tri = Triangle {
                v0: [a.x, a.y, a.z, 0.0],
                e1: edge(t[1]),
                e2: edge(t[2]),
            };
            if let Some(h) = hit_triangle(&tri, V3::of(o), V3::of(d))
                && h < t_max
                && best.is_none_or(|b| h < b.t)
            {
                best = Some(Hit {
                    t: h,
                    triangle: i as u32,
                });
            }
        }
        best
    }

    #[test]
    fn matches_brute_force() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for n in [1usize, 7, 64, 900] {
            let tris = soup(&mut rng, n);
            let bvh = Bvh::build(&tris);
            let mut sorted = bvh.order.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..n as u32).collect::<Vec<_>>());
            for _ in 0..3000 {
                let o = rng.point(8.0);
                let target = rng.point(5.0);
                let d = [target[0] - o[0], target[1] - o[1], target[2] - o[2]];
                let t_max = 0.3 + rng.next() * 1.2;
                let want = brute(&tris, o, d, t_max);
                let got = bvh.intersect(o, d, t_max);
                match (want, got) {
                    (None, None) => {}
                    (Some(w), Some(g)) => {
                        assert!(w.triangle == g.triangle || (w.t - g.t).abs() < 1e-6);
                    }
                    _ => panic!("n={n}: brute {want:?}, bvh {got:?}"),
                }
                assert_eq!(bvh.occluded(o, d, t_max), want.is_some());
            }
        }
    }

    /// Meshes placed by moves, traced through the top level and each
    /// instance's own BVH (the ray taken into its own space, `t` unchanged,
    /// as `bvh.wgsl` does), hit what the world's triangles all together hit.
    #[test]
    fn two_levels_trace_as_the_world_does() {
        let mut rng = Rng(7);
        let meshes: Vec<Vec<[[f32; 3]; 3]>> = (0..3).map(|_| soup(&mut rng, 40)).collect();
        let bvhs: Vec<Bvh> = meshes.iter().map(|m| Bvh::build(m)).collect();
        // (mesh, move) for each instance.
        let placed: Vec<(usize, [f32; 3])> = (0..12).map(|k| (k % 3, rng.point(8.0))).collect();
        let shifted = |p: [f32; 3], by: [f32; 3]| std::array::from_fn(|k| p[k] + by[k]);
        let world: Vec<[[f32; 3]; 3]> = (placed.iter())
            .flat_map(|&(m, by)| meshes[m].iter().map(move |t| t.map(|p| shifted(p, by))))
            .collect();
        let boxes: Vec<_> = (placed.iter())
            .map(|&(m, by)| {
                let (lo, hi) = bvhs[m].bounds();
                (shifted(lo, by), shifted(hi, by))
            })
            .collect();
        let tlas = Tlas::build(&boxes);
        let traced = |o: [f32; 3], d: [f32; 3], t_max: f32| -> Option<f32> {
            let (o3, d3) = (V3::of(o), V3::of(d));
            let mut best: Option<f32> = None;
            let mut stack = vec![0u32];
            while let Some(n) = stack.pop() {
                let node = &tlas.nodes[n as usize];
                if !hits_box(node, o3, V3::ONE / d3, best.unwrap_or(t_max)) {
                    continue;
                }
                if node.count == 0 {
                    stack.extend([node.a, node.a + 1]);
                    continue;
                }
                for k in node.a..node.a + node.count {
                    let (m, by) = placed[tlas.order[k as usize] as usize];
                    let own = shifted(o, by.map(|v| -v));
                    if let Some(hit) = bvhs[m].intersect(own, d, best.unwrap_or(t_max)) {
                        best = Some(hit.t);
                    }
                }
            }
            best
        };
        for _ in 0..2000 {
            let o = rng.point(14.0);
            let target = rng.point(8.0);
            let d = [target[0] - o[0], target[1] - o[1], target[2] - o[2]];
            let want = brute(&world, o, d, 1.5).map(|h| h.t);
            let got = traced(o, d, 1.5);
            match (want, got) {
                (None, None) => {}
                (Some(w), Some(g)) => assert!((w - g).abs() < 1e-4, "{w} against {g}"),
                _ => panic!("brute {want:?}, two levels {got:?}"),
            }
        }
        assert_eq!(Tlas::build(&[]).bounds(), None);
    }

    /// Three parallel cards across a ray, each two triangles: a ray keeps
    /// the product of what the cards it crosses let through, whatever their
    /// order, and nothing past an opaque one.
    #[test]
    fn transmittance_multiplies_the_cards_crossed() {
        let card = |z: f32| {
            let (a, b, c, d) = (
                [-1.0, -1.0, z],
                [1.0, -1.0, z],
                [1.0, 1.0, z],
                [-1.0, 1.0, z],
            );
            [[a, b, c], [a, c, d]]
        };
        let tris: Vec<_> = [1.0, 2.0, 3.0].into_iter().flat_map(card).collect();
        let bvh = Bvh::build(&tris);
        // Cards at z 1 and 2 pass half and a quarter in red; z 3 nothing.
        let pass = |k: u32| match k / 2 {
            0 => [0.5, 1.0, 1.0],
            1 => [0.25, 0.5, 1.0],
            _ => [0.0; 3],
        };
        let up = [0.2, 0.1, 1.0];
        let o = [0.0; 3];
        assert_eq!(bvh.transmittance(o, up, 2.5, pass), [0.125, 0.5, 1.0]);
        assert_eq!(bvh.transmittance(o, up, 0.5, pass), [1.0; 3]);
        assert_eq!(bvh.transmittance(o, up, 10.0, pass), [0.0; 3]);
        // From beyond, toward the origin: the same two cards, reversed.
        let back = [-0.2, -0.1, -1.0];
        let far = [0.5, 0.25, 2.5];
        assert_eq!(bvh.transmittance(far, back, 2.4, pass), [0.125, 0.5, 1.0]);
        // Nearly transparent cards stacked past OPAQUE count as nothing.
        let thin = |_| [OPAQUE * 0.9; 3];
        assert_eq!(bvh.transmittance(o, up, 2.5, thin), [0.0; 3]);
    }

    #[test]
    fn nodes_bound_their_triangles() {
        let mut rng = Rng(42);
        let tris = soup(&mut rng, 500);
        let bvh = Bvh::build(&tris);
        let mut stack = vec![(0u32, 0u32)];
        let mut leaves = 0;
        while let Some((n, depth)) = stack.pop() {
            assert!(depth <= MAX_DEPTH);
            let node = bvh.nodes[n as usize];
            let inside = |p: [f32; 3]| (0..3).all(|k| p[k] >= node.min[k] && p[k] <= node.max[k]);
            if node.count == 0 {
                stack.push((node.a, depth + 1));
                stack.push((node.a + 1, depth + 1));
            } else {
                leaves += node.count;
                assert!(node.count as usize <= MAX_LEAF);
                for k in node.a..node.a + node.count {
                    let t = &tris[bvh.order[k as usize] as usize];
                    assert!(t.iter().all(|&p| inside(p)));
                }
            }
        }
        assert_eq!(leaves, 500);
    }

    #[test]
    fn coincident_triangles_still_split() {
        // Many triangles with one centroid: the build halves them by order
        // rather than making one huge leaf or recursing forever.
        let tris = vec![[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]; 100];
        let bvh = Bvh::build(&tris);
        assert!(bvh.nodes.iter().all(|n| n.count as usize <= MAX_LEAF));
        assert!(bvh.occluded([0.2, 0.2, 1.0], [0.0, 0.0, -1.0], 2.0));
    }

    #[test]
    fn empty_scene_hits_nothing() {
        let bvh = Bvh::build(&[]);
        assert_eq!(bvh.nodes.len(), 1);
        assert_eq!(bvh.triangles.len(), 1);
        assert!(!bvh.occluded([0.0; 3], [0.0, 0.0, -1.0], 1e9));
    }

    #[test]
    fn no_ray_enters_the_top_level_of_no_instances() {
        // Its one leaf's box is empty (min above max), which the slab test
        // alone enters from anywhere, in any direction: the traversals would
        // then walk the zeroed instance standing in for none.
        let root = Tlas::build(&[]).nodes[0];
        let (o, d) = (V3::new(0.3, -2.0, 5.0), V3::new(0.0, 0.6, -0.8));
        let slabs = |node: &Node| {
            let (t0, t1) = ((V3::of(node.min) - o) / d, (V3::of(node.max) - o) / d);
            t0.min(t1).max_element().max(0.0) <= t0.max(t1).min_element()
        };
        assert!(slabs(&root), "the slabs alone enter it");
        assert!(!hits_box(&root, o, V3::ONE / d, f32::INFINITY));
        assert!(!hits_box(
            &root,
            V3::ZERO,
            V3::ONE / V3::new(0.0, 0.0, 1.0),
            1e9
        ));
    }
}
