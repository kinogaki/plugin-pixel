//! The scene on the GPU, in two levels: the meshes in their own space (the
//! G-buffer's vertices, and each mesh's BVH) and the instances placing them
//! (their records and the top-level BVH over them). The scene group every
//! pass binds as group 1: the meshes' nodes and triangles, the top level's
//! nodes, the volumes' BVH (nodes, triangles, records), the instances and
//! the lights. Each part is replaced only when its source changes: a camera
//! move touches none of it, an object's move the instances alone.
use super::Changed;
use super::kernel::{bind_group, storage};
use super::records::{GpuInstance, GpuLight, GpuMedium, Vertex};
use crate::render::bvh::{Bvh, Node, Tlas, Triangle};
use crate::render::vector::V3;
use crate::render::{Instance, Mesh, Scene, Surface, Volume};
use kinogaki_render_api::display::wgpu;
use std::ops::Range;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// The most volumes a frame renders (a ray's inside set is a 32-bit mask).
pub const MAX_VOLUMES: usize = 32;

/// The meshes side by side: their vertices, and their BVHs in one array of
/// nodes and one of triangles, each mesh's indices moved past the meshes
/// before it.
pub struct Meshes {
    /// Three vertices a triangle, each mesh's in a run.
    pub vertices: Vec<Vertex>,
    /// Each mesh's run of vertices, and its BVH's root node.
    pub meshes: Vec<(Range<u32>, u32)>,
    /// The BVHs' nodes.
    pub nodes: Vec<Node>,
    /// Their triangles, each with its corners' mean tint in its `w`s.
    pub triangles: Vec<Triangle>,
}

impl Meshes {
    /// `meshes` side by side.
    pub fn build(meshes: &[Arc<Mesh>]) -> Meshes {
        let mut made = Meshes {
            vertices: Vec::new(),
            meshes: Vec::new(),
            nodes: Vec::new(),
            triangles: Vec::new(),
        };
        for mesh in meshes {
            let first = made.vertices.len() as u32;
            for (t, triangle) in mesh.corners.as_chunks::<3>().0.iter().enumerate() {
                let [a, b, c] = triangle.map(V3::of);
                let face = (b - a).cross(c - a).to_array();
                for (k, &position) in triangle.iter().enumerate() {
                    let at = t * 3 + k;
                    made.vertices.push(Vertex {
                        position,
                        normal: mesh.normals.as_ref().map_or(face, |n| n[at]),
                        tint: mesh.tints.as_ref().map_or([1.0; 3], |c| c[at]),
                    });
                }
            }
            let (nodes, triangles) = (made.nodes.len() as u32, made.triangles.len() as u32);
            made.meshes.push((first..made.vertices.len() as u32, nodes));
            made.nodes.extend(mesh.bvh.nodes.iter().map(|n| Node {
                a: n.a + if n.count == 0 { nodes } else { triangles },
                ..*n
            }));
            let tinted = (mesh.bvh.triangles.iter())
                .zip(&mesh.bvh.order)
                .map(|(tri, &t)| {
                    let at = &made.vertices[(first as usize + t as usize * 3)..][..3];
                    let [r, g, b] =
                        std::array::from_fn(|k| at.iter().map(|v| v.tint[k]).sum::<f32>() / 3.0);
                    let mut tri = *tri;
                    (tri.v0[3], tri.e1[3], tri.e2[3]) = (r, g, b);
                    tri
                });
            made.triangles.extend(tinted);
        }
        made
    }
}

/// The instances as the GPU reads them: their records in the top level's
/// order, the top level, and the G-buffer's draws.
pub struct Instances {
    /// One record an instance, in the top level's leaf order.
    pub records: Vec<GpuInstance>,
    /// Each record's material index (past the scene's: the default).
    surfaces: Vec<u32>,
    /// The top level over them.
    pub tlas: Tlas,
    /// The records, by mesh: the G-buffer's instance buffer.
    pub drawn: Vec<u32>,
    /// A draw a mesh: its vertices, and its run of `drawn`.
    pub draws: Vec<(Range<u32>, Range<u32>)>,
}

impl Instances {
    /// `instances` of `meshes` (as [`Meshes::build`] laid out `bvhs`), with
    /// the materials `surfaces`.
    pub fn build(
        instances: &[Instance],
        meshes: &Meshes,
        bvhs: &[Arc<Mesh>],
        surfaces: &[Surface],
    ) -> Instances {
        let boxes: Vec<_> = (instances.iter())
            .map(|i| world_bounds(&i.to_world, bvhs[i.mesh as usize].bvh.bounds()))
            .collect();
        let tlas = Tlas::build(&boxes);
        let placed: Vec<&Instance> = tlas.order.iter().map(|&k| &instances[k as usize]).collect();
        let records = (placed.iter())
            .map(|i| {
                GpuInstance::of(
                    i,
                    meshes.meshes[i.mesh as usize].1,
                    surface(surfaces, i.surface),
                )
            })
            .collect();
        let mut drawn: Vec<u32> = (0..placed.len() as u32).collect();
        drawn.sort_by_key(|&k| placed[k as usize].mesh);
        let mut draws: Vec<(Range<u32>, Range<u32>)> = Vec::new();
        for (at, &k) in drawn.iter().enumerate() {
            let vertices = meshes.meshes[placed[k as usize].mesh as usize].0.clone();
            let at = at as u32;
            match draws.last_mut() {
                Some((v, run)) if *v == vertices => run.end = at + 1,
                _ => draws.push((vertices, at..at + 1)),
            }
        }
        Instances {
            records,
            surfaces: placed.iter().map(|i| i.surface).collect(),
            tlas,
            drawn,
            draws,
        }
    }

    /// The records with the materials `surfaces`.
    pub fn restyle(&mut self, surfaces: &[Surface]) {
        for (record, &index) in self.records.iter_mut().zip(&self.surfaces) {
            record.set_surface(surface(surfaces, index));
        }
    }
}

/// Material `index` of `surfaces`: the default past them.
fn surface(surfaces: &[Surface], index: u32) -> Surface {
    surfaces.get(index as usize).copied().unwrap_or_default()
}

/// The world bounds of own-space bounds `(lo, hi)` placed by `rows`.
fn world_bounds(rows: &[[f32; 4]; 3], (lo, hi): ([f32; 3], [f32; 3])) -> ([f32; 3], [f32; 3]) {
    let (mut min, mut max) = (V3::splat(f32::INFINITY), V3::splat(f32::NEG_INFINITY));
    for corner in 0..8 {
        let p =
            std::array::from_fn::<f32, 3, _>(|k| if corner >> k & 1 == 0 { lo[k] } else { hi[k] });
        let world = V3::of(rows.map(|r| r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3]));
        (min, max) = (min.min(world), max.max(world));
    }
    (min.to_array(), max.to_array())
}

/// The volumes ready for the GPU: their records and their bounds' BVH, each
/// stored triangle carrying its volume's index (in `v0.w`, as bits).
pub struct Media {
    /// One record a volume (one zeroed when there are none).
    pub records: Vec<GpuMedium>,
    /// Their bounds' triangles.
    pub bvh: Bvh,
    /// Volumes rendered, and how many of them cast shadows.
    pub counts: (u32, u32),
}

impl Media {
    /// The first [`MAX_VOLUMES`] of `volumes`.
    pub fn build(volumes: &[Volume]) -> Media {
        let volumes = &volumes[..volumes.len().min(MAX_VOLUMES)];
        let mut corners = Vec::new();
        let mut owners = Vec::new();
        for (index, v) in volumes.iter().enumerate() {
            for triangle in v.corners.as_chunks::<3>().0 {
                corners.push(*triangle);
                owners.push(index as u32);
            }
        }
        let mut bvh = Bvh::build(&corners);
        for (k, t) in bvh.triangles.iter_mut().enumerate() {
            if let Some(&input) = bvh.order.get(k) {
                t.v0[3] = f32::from_bits(owners[input as usize]);
            }
        }
        let mut records: Vec<GpuMedium> = volumes.iter().map(GpuMedium::of).collect();
        let casters = records.iter().filter(|r| r.flags[2] == 1).count() as u32;
        let counts = (records.len() as u32, casters);
        if records.is_empty() {
            records.push(GpuMedium::default());
        }
        Media {
            records,
            bvh,
            counts,
        }
    }
}

/// A storage (or vertex) buffer holding `data`, or one zeroed `T` when empty
/// (bindings may not be empty).
fn buffer<T: bytemuck::Pod + Default>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let fallback = [T::default()];
    let data = if data.is_empty() { &fallback[..] } else { data };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;
const VERTEX: wgpu::BufferUsages = wgpu::BufferUsages::VERTEX;

/// The scene group's bindings.
const NODES: usize = 0;
const TRIANGLES: usize = 1;
const TOP: usize = 2;
const MEDIUM_NODES: usize = 3;
const MEDIUM_TRIANGLES: usize = 4;
const MEDIA: usize = 5;
const INSTANCES: usize = 6;
const LIGHTS: usize = 7;

/// The scene's buffers and its group.
pub struct GpuScene {
    /// The meshes' vertices, three a triangle.
    pub vertices: wgpu::Buffer,
    /// The instances by mesh (`Instances::drawn`), one a drawn instance.
    pub drawn: wgpu::Buffer,
    /// The scene group's layout.
    pub layout: wgpu::BindGroupLayout,
    /// The scene group.
    pub group: wgpu::BindGroup,
    /// Its buffers in binding order.
    buffers: [wgpu::Buffer; 8],
    /// The meshes the buffers hold.
    meshes: Meshes,
    /// The instances the buffers hold.
    pub instances: Instances,
    /// Volumes rendered, and how many cast shadows.
    pub volumes: (u32, u32),
    /// Lights in the light buffer.
    pub lights: u32,
}

impl GpuScene {
    /// The buffers of an empty scene.
    pub fn new(device: &wgpu::Device) -> GpuScene {
        let stages = wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT;
        let entries: Vec<_> = (0..8).map(|b| storage(b, stages)).collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pixel scene"),
            entries: &entries,
        });
        let buffers: [wgpu::Buffer; 8] =
            std::array::from_fn(|_| buffer::<[u32; 4]>(device, "pixel empty", &[], STORAGE));
        let meshes = Meshes::build(&[]);
        let instances = Instances::build(&[], &meshes, &[], &[]);
        let mut made = GpuScene {
            vertices: buffer::<Vertex>(device, "pixel vertices", &[], VERTEX),
            drawn: buffer::<u32>(device, "pixel drawn", &[], VERTEX),
            group: Self::bind(device, &layout, &buffers),
            layout,
            buffers,
            meshes,
            instances,
            volumes: (0, 0),
            lights: 0,
        };
        made.set(device, &Scene::default(), Changed::ALL);
        made
    }

    /// Takes what `changed` says of `scene`.
    pub fn set(&mut self, device: &wgpu::Device, scene: &Scene, changed: Changed) {
        if changed.meshes {
            let m = Meshes::build(&scene.meshes);
            self.vertices = buffer(device, "pixel vertices", &m.vertices, VERTEX);
            self.buffers[NODES] = buffer(device, "pixel nodes", &m.nodes, STORAGE);
            self.buffers[TRIANGLES] = buffer(device, "pixel triangles", &m.triangles, STORAGE);
            self.meshes = m;
        }
        if changed.meshes || changed.instances {
            let i = Instances::build(
                &scene.instances,
                &self.meshes,
                &scene.meshes,
                &scene.surfaces,
            );
            self.drawn = buffer(device, "pixel drawn", &i.drawn, VERTEX);
            self.buffers[TOP] = buffer(device, "pixel top nodes", &i.tlas.nodes, STORAGE);
            self.buffers[INSTANCES] = buffer(device, "pixel instances", &i.records, STORAGE);
            self.instances = i;
        } else if changed.surfaces {
            self.instances.restyle(&scene.surfaces);
            let records = &self.instances.records;
            self.buffers[INSTANCES] = buffer(device, "pixel instances", records, STORAGE);
        }
        if changed.volumes {
            let media = Media::build(&scene.volumes);
            self.buffers[MEDIUM_NODES] =
                buffer(device, "pixel medium nodes", &media.bvh.nodes, STORAGE);
            self.buffers[MEDIUM_TRIANGLES] = buffer(
                device,
                "pixel medium triangles",
                &media.bvh.triangles,
                STORAGE,
            );
            self.buffers[MEDIA] = buffer(device, "pixel media", &media.records, STORAGE);
            self.volumes = media.counts;
        }
        if changed.lights {
            let packed: Vec<GpuLight> = scene.lights.iter().map(GpuLight::of).collect();
            self.buffers[LIGHTS] = buffer(device, "pixel lights", &packed, STORAGE);
            self.lights = packed.len() as u32;
        }
        if changed.any() {
            self.group = Self::bind(device, &self.layout, &self.buffers);
        }
    }

    /// The bounds of what is drawn (none when nothing is).
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        self.instances.tlas.bounds()
    }

    /// The bounds' diagonal (0 when nothing is drawn).
    pub fn extent(&self) -> f32 {
        self.bounds()
            .map_or(0.0, |(lo, hi)| (V3::of(hi) - V3::of(lo)).length())
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffers: &[wgpu::Buffer; 8],
    ) -> wgpu::BindGroup {
        let all: Vec<_> = buffers
            .iter()
            .map(wgpu::Buffer::as_entire_binding)
            .collect();
        bind_group(device, "pixel scene", layout, &all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(z: f32) -> Mesh {
        let [a, b, c, d] = [[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]];
        Mesh::new(vec![a, b, c, a, c, d], None, Some(vec![[0.5; 3]; 6]))
    }

    /// Placed `dx` along X, in material `surface`.
    fn instance(mesh: u32, dx: f32, surface: u32) -> Instance {
        let rows = |x| {
            [
                [1.0, 0.0, 0.0, x],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        };
        Instance {
            mesh,
            to_world: rows(dx),
            to_own: Some(rows(-dx)),
            surface,
            id: 7,
        }
    }

    #[test]
    fn meshes_lie_side_by_side_and_instances_draw_by_mesh() {
        let meshes = [Arc::new(quad(0.0)), Arc::new(quad(1.0))];
        let m = Meshes::build(&meshes);
        assert_eq!(m.vertices.len(), 12);
        // Flat normals face +Z; each triangle's mean tint rides in its w's.
        assert_eq!(m.vertices[0].normal, [0.0, 0.0, 1.0]);
        assert!(
            m.triangles
                .iter()
                .all(|t| [t.v0[3], t.e1[3], t.e2[3]] == [0.5; 3])
        );
        // The second mesh's nodes and triangles are moved past the first's.
        let (first, second) = (&m.meshes[0], &m.meshes[1]);
        assert_eq!((first.0.clone(), second.0.clone()), (0..6, 6..12));
        let root = m.nodes[second.1 as usize];
        assert!(root.count > 0 && root.a >= meshes[0].bvh.triangles.len() as u32);
        let surfaces = [
            crate::render::Surface {
                color: [0.2, 0.3, 0.4],
                translucency: 0.5,
                translucency_color: [1.0, 0.5, 0.0],
                unlit: false,
            },
            crate::render::Surface {
                unlit: true,
                ..Default::default()
            },
        ];
        let placed = [
            instance(1, 0.0, 0),
            instance(0, 2.0, 3),
            instance(1, 4.0, 1),
        ];
        let i = Instances::build(&placed, &m, &meshes, &surfaces);
        // Two draws: mesh 0 once, mesh 1 twice.
        let runs: Vec<_> = i.draws.iter().map(|(v, r)| (v.clone(), r.len())).collect();
        assert_eq!(runs, [(0..6, 1), (6..12, 2)]);
        assert_eq!(i.tlas.bounds(), Some(([0.0, 0.0, 0.0], [5.0, 1.0, 1.0])));
        let of = |dx: f32| i.records.iter().find(|r| r.to_world[0][3] == dx).unwrap();
        assert_eq!(of(0.0).albedo, [0.2, 0.3, 0.4, 0.0]);
        assert_eq!(of(0.0).transmit, [0.5, 0.25, 0.0, 0.0]);
        assert_eq!(of(0.0).id, [7, second.1, 0, 0]);
        // An unlit one says so in its transmittance's w; a material past the
        // table is the default surface.
        assert_eq!(of(4.0).transmit, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(of(2.0).albedo, [0.8, 0.8, 0.8, 0.0]);
    }

    #[test]
    fn media_carry_their_volumes_index() {
        let box_of = |shadows| Volume {
            corners: quad(0.0).corners,
            to_own: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            own_bounds: ([0.0; 3], [1.0; 3]),
            scale: [1.0; 3],
            medium: crate::render::Medium {
                density: 1.0,
                scatter_color: [1.0; 3],
                anisotropy: 0.0,
                emission: [0.0; 3],
                noise_amount: 0.0,
                noise_frequency: 1.0,
                noise_octaves: 1,
                noise_offset: [0.0; 3],
                noise_seed: 0,
                edge_falloff: 0.0,
                shadows,
            },
        };
        let media = Media::build(&[box_of(false), box_of(true)]);
        assert_eq!(media.counts, (2, 1));
        let owners: Vec<u32> = media
            .bvh
            .triangles
            .iter()
            .map(|t| t.v0[3].to_bits())
            .collect();
        assert_eq!(owners.iter().filter(|&&o| o == 1).count(), 2);
        assert_eq!(Media::build(&[]).records.len(), 1);
    }
}
