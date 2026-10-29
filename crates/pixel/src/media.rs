//! Volumes on the GPU: their bounds' triangles in a BVH of their own and a
//! record each ([`GpuMedium`], `Medium` in shaders/medium.wgsl) holding its
//! medium and the frame of its bounding box. Built when the volumes or
//! their materials change; small (a box is twelve triangles).
//!
//! A ray finds a volume's intervals by its crossings of that volume's
//! triangles: each stored triangle carries its volume's index (in `v0.w`,
//! as bits), and a closed mesh is crossed an even number of times, so the
//! crossings' parity says where the ray is inside. Surfaces never see
//! these triangles: they are not in the surface BVH or the G-buffer.

use crate::math::{Vec3, Vec4};
use bytemuck::{Pod, Zeroable};

use crate::bvh::Bvh;
use crate::geometry::matrix;
use crate::scene::Volume;
use crate::volume::{PixelVolume, edge_share, noise_share};

/// The most volumes a frame renders (a ray's inside set is a 32-bit mask).
pub const MAX_VOLUMES: usize = 32;

/// A volume as the passes read it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuMedium {
    /// World to the mesh's own space: three rows.
    pub to_local: [[f32; 4]; 3],
    /// xyz: the bounding box's minimum, own space; w: the edge falloff.
    pub lo: [f32; 4],
    /// xyz: its maximum; w: the density.
    pub hi: [f32; 4],
    /// xyz: scene units a unit of own space spans along each axis; w: the
    /// anisotropy.
    pub scale: [f32; 4],
    /// rgb: the scatter color; w: the noise amount.
    pub scatter: [f32; 4],
    /// rgb: emission × its color; w: the noise frequency.
    pub emission: [f32; 4],
    /// xyz: the noise offset.
    pub offset: [f32; 4],
    /// x: noise octaves; y: the noise seed; z: 1 when it casts shadows.
    pub flags: [u32; 4],
}

/// The volumes, ready for the GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct Media {
    /// One record a volume (one zeroed when there are none).
    pub records: Vec<GpuMedium>,
    /// Their bounds' triangles, each carrying its volume's index.
    pub bvh: Bvh,
    /// Volumes rendered.
    pub count: u32,
    /// How many of them cast shadows.
    pub casters: u32,
}

fn f32x4(v: Vec3, w: f32) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, w]
}

impl Media {
    /// The visible volumes' records and BVH (the first [`MAX_VOLUMES`]).
    pub fn build(volumes: &[Volume]) -> Media {
        let mut records = Vec::new();
        let mut corners = Vec::new();
        let mut owners = Vec::new();
        for v in volumes.iter().filter(|v| v.mesh.visible).take(MAX_VOLUMES) {
            let index = records.len() as u32;
            let to_world = matrix(&v.mesh.transform);
            let world: Vec<Vec3> = v
                .mesh
                .points
                .iter()
                .map(|p| {
                    to_world.transform_point3(Vec3::new(p[0].into(), p[1].into(), p[2].into()))
                })
                .collect();
            for tri in v.mesh.triangles.iter() {
                let [Some(a), Some(b), Some(c)] = tri.map(|i| world.get(i as usize)) else {
                    continue;
                };
                corners.push([a, b, c].map(|p| [p.x as f32, p.y as f32, p.z as f32]));
                owners.push(index);
            }
            records.push(record(v, to_world));
        }
        let mut bvh = Bvh::build(&corners);
        for (k, t) in bvh.triangles.iter_mut().enumerate() {
            if let Some(&input) = bvh.order.get(k) {
                t.v0[3] = f32::from_bits(owners[input as usize]);
            }
        }
        let count = records.len() as u32;
        let casters = records.iter().filter(|r| r.flags[2] == 1).count() as u32;
        if records.is_empty() {
            records.push(GpuMedium::default());
        }
        Media {
            records,
            bvh,
            count,
            casters,
        }
    }
}

/// Volume `v`'s record: its medium, clamped, and its bounding box's frame.
fn record(v: &Volume, to_world: crate::math::Mat4) -> GpuMedium {
    let m: PixelVolume = v.medium.clamped();
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for p in v.mesh.points.iter() {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let inverse = to_world.inverse();
    let row = |r: usize| {
        let c = |i: usize| inverse.col(i)[r] as f32;
        [c(0), c(1), c(2), c(3)]
    };
    let axis = |i: usize| to_world.col(i).truncate().length() as f32;
    let e = m.emission_color.map(|c| c * m.emission);
    GpuMedium {
        to_local: [row(0), row(1), row(2)],
        lo: [lo[0], lo[1], lo[2], m.edge_falloff],
        hi: [hi[0], hi[1], hi[2], m.density],
        scale: f32x4(
            Vec3::new(axis(0).into(), axis(1).into(), axis(2).into()),
            m.anisotropy,
        ),
        scatter: [
            m.scatter_color[0],
            m.scatter_color[1],
            m.scatter_color[2],
            m.noise_amount,
        ],
        emission: [e[0], e[1], e[2], m.noise_frequency],
        offset: [m.noise_offset[0], m.noise_offset[1], m.noise_offset[2], 0.0],
        flags: [m.noise_octaves, m.noise_seed, u32::from(m.shadows), 0],
    }
}

/// The density of `r`'s medium at world point `x` (which the caller knows
/// to be inside it): its density × the edge falloff's share × the noise's.
pub fn density(r: &GpuMedium, x: [f32; 3]) -> f32 {
    let p = Vec4::new(x[0].into(), x[1].into(), x[2].into(), 1.0);
    let local = [0, 1, 2].map(|k| {
        let row = r.to_local[k];
        (f64::from(row[0]) * p.x
            + f64::from(row[1]) * p.y
            + f64::from(row[2]) * p.z
            + f64::from(row[3])) as f32
    });
    let inside = (0..3)
        .map(|k| (local[k] - r.lo[k]).min(r.hi[k] - local[k]) * r.scale[k])
        .fold(f32::INFINITY, f32::min);
    let medium = PixelVolume {
        noise_amount: r.scatter[3],
        noise_frequency: r.emission[3],
        noise_octaves: r.flags[0],
        noise_offset: [r.offset[0], r.offset[1], r.offset[2]],
        noise_seed: r.flags[1],
        ..PixelVolume::default()
    };
    r.hi[3] * edge_share(inside, r.lo[3]) * noise_share(&medium, x)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::scene::{IDENTITY, Mesh};

    /// A unit cube from 0 to 1, as twelve triangles, placed by `transform`.
    pub fn cube(transform: crate::Matrix) -> Mesh {
        let p = |x: f32, y: f32, z: f32| [x, y, z];
        let points = vec![
            p(0.0, 0.0, 0.0),
            p(1.0, 0.0, 0.0),
            p(1.0, 1.0, 0.0),
            p(0.0, 1.0, 0.0),
            p(0.0, 0.0, 1.0),
            p(1.0, 0.0, 1.0),
            p(1.0, 1.0, 1.0),
            p(0.0, 1.0, 1.0),
        ];
        let quads = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [3, 7, 6, 2],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
        ];
        let triangles: Vec<[u32; 3]> = quads
            .iter()
            .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
            .collect();
        Mesh {
            points: Arc::from(points),
            triangles: Arc::from(triangles),
            normals: None,
            tints: None,
            transform,
            material: 0,
            id: 1,
            visible: true,
        }
    }

    #[test]
    fn records_carry_their_volume_and_the_bvh_its_index() {
        let mut stretched = IDENTITY;
        stretched[0][0] = 4.0;
        stretched[3] = [10.0, 0.0, 0.0, 1.0];
        let volumes = [
            Volume {
                mesh: cube(IDENTITY),
                medium: PixelVolume::default(),
            },
            Volume {
                mesh: cube(stretched),
                medium: PixelVolume {
                    shadows: true,
                    edge_falloff: 1.0,
                    ..PixelVolume::default()
                },
            },
        ];
        let media = Media::build(&volumes);
        assert_eq!((media.count, media.casters), (2, 1));
        assert_eq!(media.bvh.triangles.len(), 24);
        for (k, t) in media.bvh.triangles.iter().enumerate() {
            let owner = media.bvh.order[k] / 12;
            assert_eq!(t.v0[3].to_bits(), owner);
        }
        let r = &media.records[1];
        assert_eq!(r.scale[0], 4.0);
        // The stretched box spans x 10..14: its middle is 2 units in along
        // x, but half a unit from the y and z faces.
        assert!((density(r, [12.0, 0.5, 0.5]) - 0.5 * edge_share(0.5, 1.0)).abs() < 1e-6);
        assert_eq!(density(&media.records[0], [0.5; 3]), 0.5);
        let none = Media::build(&[]);
        assert_eq!((none.count, none.records.len()), (0, 1));
    }
}
