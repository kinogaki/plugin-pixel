//! World-space geometry: the meshes flattened into the G-buffer's vertices,
//! the BVH's triangles and what a ray hitting one shades with ([`Surface`]).
//! Built when geometry changes; a camera move touches none of it.

use crate::math::{Mat4, Vec3, Vec4};
use bytemuck::{Pod, Zeroable};

use crate::bvh::Bvh;
use crate::scene::{Matrix, Mesh, PixelSurface};

/// A G-buffer vertex: one per triangle corner, in world space.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    /// World position.
    pub position: [f32; 3],
    /// Index into the object table.
    pub object: u32,
    /// World shading normal (not unit).
    pub normal: [f32; 3],
    /// What the material's Color is multiplied by (the mesh's color
    /// attribute, linear ACEScg; white without one).
    pub tint: [f32; 3],
}

/// An object table entry (`Object` in shaders/common.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Object {
    /// The material's color (ACEScg), w unused.
    pub albedo: [f32; 4],
    /// The material's transmittance ([`PixelSurface::transmittance`]), w
    /// unused.
    pub transmit: [f32; 4],
    /// x: the object id.
    pub id: [u32; 4],
}

/// What a ray hitting a BVH triangle shades with (`Surface` in
/// shaders/bounce.wgsl): its object and its corners' mean tint, one a
/// stored triangle, in the BVH's order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Surface {
    /// The mean of its corners' tints.
    pub tint: [f32; 3],
    /// Index into the object table.
    pub object: u32,
}

/// The visible meshes in world space.
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    /// Three vertices a triangle.
    pub vertices: Vec<Vertex>,
    /// Each visible mesh's material index and id, by object index.
    pub objects: Vec<(u32, u32)>,
    /// The BVH over the same triangles, in the same order.
    pub bvh: Bvh,
    /// Each of the BVH's stored triangles' surface, in its order (one
    /// for the empty scene's degenerate triangle).
    pub surfaces: Vec<Surface>,
}

pub(crate) fn matrix(m: &Matrix) -> Mat4 {
    let c = |i: usize| {
        Vec4::new(
            m[i][0] as f64,
            m[i][1] as f64,
            m[i][2] as f64,
            m[i][3] as f64,
        )
    };
    Mat4::from_cols(c(0), c(1), c(2), c(3))
}

fn f32x3(v: Vec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

fn vec3(p: [f32; 3]) -> Vec3 {
    Vec3::new(p[0] as f64, p[1] as f64, p[2] as f64)
}

impl Geometry {
    /// The visible meshes, flattened. Triangles naming points a mesh does
    /// not have are skipped.
    pub fn build(meshes: &[Mesh]) -> Geometry {
        let mut vertices = Vec::new();
        let mut objects = Vec::new();
        let mut corners = Vec::new();
        for mesh in meshes.iter().filter(|m| m.visible) {
            let object = objects.len() as u32;
            objects.push((mesh.material, mesh.id));
            let to_world = matrix(&mesh.transform);
            let normal_matrix = to_world.inverse().transpose();
            let world: Vec<Vec3> = mesh
                .points
                .iter()
                .map(|&p| to_world.transform_point3(vec3(p)))
                .collect();
            let corners_of = |c: &Option<std::sync::Arc<[[f32; 3]]>>| {
                c.clone().filter(|c| c.len() == mesh.triangles.len() * 3)
            };
            let shading = corners_of(&mesh.normals);
            let tints = corners_of(&mesh.tints);
            for (t, tri) in mesh.triangles.iter().enumerate() {
                let [Some(&a), Some(&b), Some(&c)] = tri.map(|i| world.get(i as usize)) else {
                    continue;
                };
                let p = [a, b, c];
                let face = (p[1] - p[0]).cross(p[2] - p[0]);
                for k in 0..3 {
                    let n = match &shading {
                        Some(n) => normal_matrix.transform_vector3(vec3(n[t * 3 + k])),
                        None => face,
                    };
                    vertices.push(Vertex {
                        position: f32x3(p[k]),
                        object,
                        normal: f32x3(n),
                        tint: tints.as_ref().map_or([1.0; 3], |c| c[t * 3 + k]),
                    });
                }
                corners.push([f32x3(p[0]), f32x3(p[1]), f32x3(p[2])]);
            }
        }
        let bvh = Bvh::build(&corners);
        let surfaces = surfaces(&vertices, &bvh);
        Geometry {
            vertices,
            objects,
            bvh,
            surfaces,
        }
    }

    /// The object table for `materials`: a material index past the end
    /// takes the default Pixel Surface.
    pub fn object_table(&self, materials: &[PixelSurface]) -> Vec<Object> {
        self.objects
            .iter()
            .map(|&(material, id)| {
                let m = materials
                    .get(material as usize)
                    .cloned()
                    .unwrap_or_default();
                let t = m.transmittance();
                Object {
                    albedo: [m.color[0], m.color[1], m.color[2], 0.0],
                    transmit: [t[0], t[1], t[2], 0.0],
                    id: [id, 0, 0, 0],
                }
            })
            .collect()
    }

    /// The scene's bounds (none when empty).
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        (!self.vertices.is_empty()).then(|| self.bvh.bounds())
    }

    /// The scene's bounds' diagonal (0 when empty).
    pub fn extent(&self) -> f32 {
        let Some((lo, hi)) = self.bounds() else {
            return 0.0;
        };
        let d = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }
}

/// The surfaces of `bvh`'s stored triangles: each one's object and mean
/// tint, from its three vertices.
fn surfaces(vertices: &[Vertex], bvh: &Bvh) -> Vec<Surface> {
    if bvh.order.is_empty() {
        return vec![Surface::default()];
    }
    bvh.order
        .iter()
        .map(|&t| {
            let corners = &vertices[t as usize * 3..t as usize * 3 + 3];
            let mean = |k: usize| corners.iter().map(|v| v.tint[k]).sum::<f32>() / 3.0;
            Surface {
                tint: [mean(0), mean(1), mean(2)],
                object: corners[0].object,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::scene::IDENTITY;

    fn quad(transform: Matrix, visible: bool) -> Mesh {
        Mesh {
            points: Arc::from(vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
            triangles: Arc::from(vec![[0, 1, 2], [0, 2, 3], [0, 2, 9]]),
            normals: None,
            tints: None,
            transform,
            material: 1,
            id: 7,
            visible,
        }
    }

    #[test]
    fn meshes_flatten_to_world_space() {
        let mut moved = IDENTITY;
        moved[3] = [0.0, 0.0, 2.0, 1.0];
        let g = Geometry::build(&[quad(moved, true), quad(IDENTITY, false)]);
        // The out-of-range triangle is skipped; the hidden mesh is gone.
        assert_eq!(g.vertices.len(), 6);
        assert_eq!(g.objects, vec![(1, 7)]);
        assert!(g.vertices.iter().all(|v| v.position[2] == 2.0));
        assert!(g.vertices.iter().all(|v| v.normal[2] > 0.0));
        assert!(g.vertices.iter().all(|v| v.tint == [1.0; 3]));
        assert!(g.bvh.occluded([0.5, 0.5, 3.0], [0.0, 0.0, -1.0], 2.0));
        assert!(!g.bvh.occluded([0.5, 0.5, 3.0], [0.0, 0.0, -1.0], 0.9));
        let table = g.object_table(&[PixelSurface::default()]);
        // Tints reach the vertices, three a triangle.
        let mut tinted = quad(IDENTITY, true);
        tinted.triangles = Arc::from(vec![[0, 1, 2]]);
        tinted.tints = Some(Arc::from(vec![[0.5, 0.25, 1.0]; 3]));
        let g = Geometry::build(&[tinted]);
        assert!(g.vertices.iter().all(|v| v.tint == [0.5, 0.25, 1.0]));
        assert_eq!(table[0].transmit, [0.0; 4], "opaque");
        assert_eq!(table[0].albedo[0], PixelSurface::default().color[0]);
        assert_eq!(table[0].id[0], 7);
    }

    #[test]
    fn surfaces_follow_the_bvh_order_with_mean_tints() {
        let tinted = |x: f32, tint: [f32; 3], material| {
            let mut m = quad(IDENTITY, true);
            m.transform[3] = [x, 0.0, 0.0, 1.0];
            m.triangles = Arc::from(vec![[0, 1, 2]]);
            m.tints = Some(Arc::from(vec![tint, tint, [0.0; 3]]));
            m.material = material;
            m
        };
        let g = Geometry::build(&[
            tinted(0.0, [0.3, 0.6, 0.9], 0),
            tinted(5.0, [0.9, 0.0, 0.3], 1),
        ]);
        assert_eq!(g.surfaces.len(), g.bvh.triangles.len());
        for (k, s) in g.surfaces.iter().enumerate() {
            let t = g.bvh.order[k] as usize;
            assert_eq!(s.object, t as u32);
            let want = [[0.2, 0.4, 0.6], [0.6, 0.0, 0.2]][t];
            assert!((0..3).all(|i| (s.tint[i] - want[i]).abs() < 1e-6), "{s:?}");
        }
        let table = g.object_table(&[PixelSurface {
            translucency: 0.5,
            translucency_color: [1.0, 0.5, 0.0],
            ..PixelSurface::default()
        }]);
        assert_eq!(table[0].transmit, [0.5, 0.25, 0.0, 0.0]);
        assert_eq!(table[1].transmit, [0.0; 4], "the default surface");
        assert_eq!(Geometry::build(&[]).surfaces, vec![Surface::default()]);
    }
}
