//! A mesh as Pixel renders it: indexed, each distinct corner (a point and
//! what it holds there) one vertex, so a smooth surface keeps about one
//! vertex a point where three corners a triangle cost three.
use super::bvh::Bvh;
use super::vector::V3;

/// Triangles of one material in their own space: their vertices, three a
/// triangle, and the hierarchy rays trace them by.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    /// Each vertex's position.
    pub positions: Vec<[f32; 3]>,
    /// Each vertex's shading normal: as held, else its triangle's own (flat).
    pub normals: Vec<[f32; 3]>,
    /// What each vertex's albedo is multiplied by (the material's color
    /// attribute; none: white).
    pub tints: Option<Vec<[f32; 3]>>,
    /// Each vertex's texture coordinates, which the material's texture
    /// reads (none: it has none).
    pub uvs: Option<Vec<[f32; 2]>>,
    /// Three vertices a triangle.
    pub triangles: Vec<[u32; 3]>,
    /// The hierarchy over its triangles.
    pub bvh: Bvh,
}

/// What a mesh holds at each corner beside its position, three corners a
/// triangle.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Corners {
    /// Shading normals (none: flat).
    pub normals: Option<Vec<[f32; 3]>>,
    /// What the albedo is multiplied by (none: white).
    pub tints: Option<Vec<[f32; 3]>>,
    /// Texture coordinates (none: none).
    pub uvs: Option<Vec<[f32; 2]>>,
}

impl Mesh {
    /// The mesh of `triangles` over `points`, what it holds at their corners
    /// kept only where it matches them: corners on one point holding the
    /// same values share a vertex. Its hierarchy built.
    pub fn new(points: &[[f32; 3]], triangles: &[[u32; 3]], held: Corners) -> Mesh {
        let corners = triangles.len() * 3;
        let normals = held.normals.filter(|c| c.len() == corners);
        let tints = held.tints.filter(|c| c.len() == corners);
        let uvs = held.uvs.filter(|c| c.len() == corners);
        let mut mesh = Mesh {
            positions: Vec::new(),
            normals: Vec::new(),
            tints: tints.as_ref().map(|_| Vec::new()),
            uvs: uvs.as_ref().map(|_| Vec::new()),
            triangles: Vec::with_capacity(triangles.len()),
            bvh: Bvh::build(&[]),
        };
        // The vertices made on each point: the newest, and each one's next.
        let mut newest = vec![u32::MAX; points.len()];
        let mut next: Vec<u32> = Vec::new();
        for (t, triangle) in triangles.iter().enumerate() {
            let flat = || {
                let [a, b, c] = triangle.map(|p| V3::of(points[p as usize]));
                (b - a).cross(c - a).unit_or(V3::ZERO).to_array()
            };
            let face = normals.is_none().then(flat);
            let vertices = std::array::from_fn(|k| {
                let at = t * 3 + k;
                let corner = Corner {
                    position: points[triangle[k] as usize],
                    normal: face.unwrap_or_else(|| normals.as_ref().map_or([0.0; 3], |n| n[at])),
                    tint: tints.as_ref().map(|c| c[at]),
                    uv: uvs.as_ref().map(|c| c[at]),
                };
                let point = triangle[k] as usize;
                let mut v = newest[point];
                while v != u32::MAX && !mesh.holds(v as usize, &corner) {
                    v = next[v as usize];
                }
                if v == u32::MAX {
                    v = mesh.push(&corner);
                    next.push(newest[point]);
                    newest[point] = v;
                }
                v
            });
            mesh.triangles.push(vertices);
        }
        let traced: Vec<[[f32; 3]; 3]> = (mesh.triangles.iter())
            .map(|t| t.map(|v| mesh.positions[v as usize]))
            .collect();
        mesh.bvh = Bvh::build(&traced);
        mesh
    }

    /// Whether vertex `v` is `corner`.
    fn holds(&self, v: usize, corner: &Corner) -> bool {
        self.positions[v] == corner.position
            && self.normals[v] == corner.normal
            && self.tints.as_ref().map(|c| c[v]) == corner.tint
            && self.uvs.as_ref().map(|c| c[v]) == corner.uv
    }

    /// `corner` as a new vertex; its index.
    fn push(&mut self, corner: &Corner) -> u32 {
        self.positions.push(corner.position);
        self.normals.push(corner.normal);
        if let (Some(tints), Some(tint)) = (&mut self.tints, corner.tint) {
            tints.push(tint);
        }
        if let (Some(uvs), Some(uv)) = (&mut self.uvs, corner.uv) {
            uvs.push(uv);
        }
        self.positions.len() as u32 - 1
    }
}

/// One triangle corner: its position and what it holds there.
struct Corner {
    position: [f32; 3],
    normal: [f32; 3],
    tint: Option<[f32; 3]>,
    uv: Option<[f32; 2]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit square of two triangles over four points.
    const SQUARE: [[f32; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    const HALVES: [[u32; 3]; 2] = [[0, 1, 2], [0, 2, 3]];

    #[test]
    fn corners_holding_the_same_share_a_vertex() {
        // Flat and smooth alike: the diagonal's two points are shared.
        let flat = Mesh::new(&SQUARE, &HALVES, Corners::default());
        assert_eq!(flat.positions.len(), 4);
        assert_eq!(flat.triangles, [[0, 1, 2], [0, 2, 3]]);
        assert!(flat.normals.iter().all(|n| *n == [0.0, 0.0, 1.0]));
        let up = Some(vec![[0.0, 0.0, 1.0]; 6]);
        let tints = Some(vec![[0.5; 3]; 6]);
        let smooth = Mesh::new(
            &SQUARE,
            &HALVES,
            Corners {
                normals: up.clone(),
                tints,
                uvs: None,
            },
        );
        assert_eq!(
            (smooth.positions.len(), smooth.tints.map(|t| t.len())),
            (4, Some(4))
        );
        // A seam: the second triangle's coordinates differ where they meet.
        let uv = |p: [f32; 3]| [p[0], p[1]];
        let mut uvs: Vec<[f32; 2]> = HALVES
            .iter()
            .flatten()
            .map(|&p| uv(SQUARE[p as usize]))
            .collect();
        uvs[3] = [9.0, 9.0];
        uvs[4] = [9.0, 9.0];
        let seamed = Mesh::new(
            &SQUARE,
            &HALVES,
            Corners {
                normals: up,
                tints: None,
                uvs: Some(uvs),
            },
        );
        assert_eq!(seamed.positions.len(), 6);
        assert_eq!(seamed.triangles[1], [3, 4, 5]);
        assert_eq!(seamed.uvs.as_ref().map(|u| u[3]), Some([9.0, 9.0]));
    }

    #[test]
    fn a_folded_square_keeps_each_halfs_flat_normal() {
        let mut folded = SQUARE;
        folded[3][2] = 1.0;
        let mesh = Mesh::new(&folded, &HALVES, Corners::default());
        // The fold's points carry two normals, the others one.
        assert_eq!(mesh.positions.len(), 6);
        let up = |v: u32| mesh.normals[v as usize];
        assert_eq!(up(mesh.triangles[0][0]), [0.0, 0.0, 1.0]);
        assert_ne!(up(mesh.triangles[1][0]), [0.0, 0.0, 1.0]);
        // What it holds that doesn't match its corners is dropped.
        let short = Corners {
            tints: Some(vec![[1.0; 3]; 5]),
            ..Corners::default()
        };
        assert_eq!(Mesh::new(&SQUARE, &HALVES, short).tints, None);
    }

    #[test]
    fn its_hierarchy_traces_its_triangles() {
        let mesh = Mesh::new(&SQUARE, &HALVES, Corners::default());
        assert_eq!(mesh.bvh.triangles.len(), 2);
        assert_eq!(mesh.bvh.bounds(), ([0.0; 3], [1.0, 1.0, 0.0]));
    }
}
