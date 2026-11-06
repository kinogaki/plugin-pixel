//! The scene on the GPU: the G-buffer's vertices, and the scene group every
//! pass binds as group 1: the surface BVH (nodes, triangles, their
//! surfaces), the volumes' BVH (nodes, triangles, records), the object table
//! and the lights. Each part is replaced only when its source changes: a
//! camera move touches none of it.
use super::kernel::{bind_group, storage};
use super::records::{GpuLight, GpuMedium, Object, Surface, Vertex};
use crate::render::bvh::Bvh;
use crate::render::vector::V3;
use crate::render::{Light, Mesh, Scene, Volume};
use kinogaki_render_api::display::wgpu;
use wgpu::util::DeviceExt;

/// The most volumes a frame renders (a ray's inside set is a 32-bit mask).
pub const MAX_VOLUMES: usize = 32;

/// The surfaces in world space, flattened.
pub struct Geometry {
    /// Three vertices a triangle.
    pub vertices: Vec<Vertex>,
    /// Each mesh's material index and id, by object index.
    pub objects: Vec<(u32, u32)>,
    /// The BVH over the same triangles.
    pub bvh: Bvh,
    /// Each stored triangle's surface, in the BVH's order.
    pub surfaces: Vec<Surface>,
}

impl Geometry {
    /// `meshes` flattened; a mesh whose corners aren't whole triangles, or
    /// whose normals or tints don't match them, keeps what fits.
    pub fn build(meshes: &[Mesh]) -> Geometry {
        let mut vertices = Vec::new();
        let mut objects = Vec::new();
        let mut corners = Vec::new();
        for mesh in meshes {
            let object = objects.len() as u32;
            objects.push((mesh.surface, mesh.id));
            let per_corner = |c: &Option<Vec<[f32; 3]>>| {
                c.as_ref()
                    .filter(|c| c.len() == mesh.corners.len())
                    .cloned()
            };
            let (normals, tints) = (per_corner(&mesh.normals), per_corner(&mesh.tints));
            for (t, triangle) in mesh.corners.as_chunks::<3>().0.iter().enumerate() {
                let [a, b, c] = triangle.map(V3::of);
                let face = (b - a).cross(c - a).to_array();
                for (k, &position) in triangle.iter().enumerate() {
                    let at = t * 3 + k;
                    vertices.push(Vertex {
                        position,
                        object,
                        normal: normals.as_ref().map_or(face, |n| n[at]),
                        tint: tints.as_ref().map_or([1.0; 3], |c| c[at]),
                    });
                }
                corners.push(*triangle);
            }
        }
        let bvh = Bvh::build(&corners);
        let surfaces = match bvh.order.is_empty() {
            true => vec![Surface::default()],
            false => (bvh.order.iter())
                .map(|&t| {
                    let at = &vertices[t as usize * 3..][..3];
                    let mean = |k: usize| at.iter().map(|v| v.tint[k]).sum::<f32>() / 3.0;
                    Surface {
                        tint: [mean(0), mean(1), mean(2)],
                        object: at[0].object,
                    }
                })
                .collect(),
        };
        Geometry {
            vertices,
            objects,
            bvh,
            surfaces,
        }
    }

    /// The object table for the scene's materials (an index past them takes
    /// the default surface).
    pub fn objects(&self, surfaces: &[crate::render::Surface]) -> Vec<Object> {
        (self.objects.iter())
            .map(|&(surface, id)| {
                let m = surfaces.get(surface as usize).copied().unwrap_or_default();
                Object {
                    albedo: V3::of(m.color).with_w(0.0),
                    transmit: V3::of(m.transmittance()).with_w(0.0),
                    id: [id, 0, 0, 0],
                }
            })
            .collect()
    }

    /// The bounds (none when empty).
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        (!self.vertices.is_empty()).then(|| self.bvh.bounds())
    }

    /// The bounds' diagonal (0 when empty).
    pub fn extent(&self) -> f32 {
        self.bounds()
            .map_or(0.0, |(lo, hi)| (V3::of(hi) - V3::of(lo)).length())
    }
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

/// The scene's buffers and its group.
pub struct GpuScene {
    /// G-buffer vertices, three a triangle.
    pub vertices: wgpu::Buffer,
    /// Vertices to draw.
    pub vertex_count: u32,
    /// The scene group's layout.
    pub layout: wgpu::BindGroupLayout,
    /// The scene group.
    pub group: wgpu::BindGroup,
    /// Its buffers in binding order: surface nodes, triangles and surfaces,
    /// volume nodes, triangles and records, objects, lights.
    buffers: [wgpu::Buffer; 8],
    /// The geometry the buffers hold.
    pub geometry: Geometry,
    /// Volumes rendered, and how many cast shadows.
    pub volumes: (u32, u32),
    /// Lights in the light buffer.
    pub lights: u32,
}

impl GpuScene {
    /// The buffers of `scene`.
    pub fn new(device: &wgpu::Device, scene: &Scene) -> GpuScene {
        let stages = wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT;
        let entries: Vec<_> = (0..8).map(|b| storage(b, stages)).collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pixel scene"),
            entries: &entries,
        });
        let buffers: [wgpu::Buffer; 8] =
            std::array::from_fn(|_| buffer::<[u32; 4]>(device, "pixel empty", &[], STORAGE));
        let all: Vec<_> = buffers
            .iter()
            .map(wgpu::Buffer::as_entire_binding)
            .collect();
        let geometry = Geometry::build(&[]);
        let mut made = GpuScene {
            vertices: buffer::<Vertex>(device, "pixel vertices", &[], wgpu::BufferUsages::VERTEX),
            vertex_count: 0,
            group: bind_group(device, "pixel scene", &layout, &all),
            layout,
            buffers,
            geometry,
            volumes: (0, 0),
            lights: 0,
        };
        made.set_geometry(device, scene);
        made.set_volumes(device, &scene.volumes);
        made.set_lights(device, &scene.lights);
        made
    }

    /// New meshes or materials: the vertices, the surface BVH and the object
    /// table.
    pub fn set_geometry(&mut self, device: &wgpu::Device, scene: &Scene) {
        let g = Geometry::build(&scene.meshes);
        self.vertices = buffer(
            device,
            "pixel vertices",
            &g.vertices,
            wgpu::BufferUsages::VERTEX,
        );
        self.vertex_count = g.vertices.len() as u32;
        self.buffers[0] = buffer(device, "pixel nodes", &g.bvh.nodes, STORAGE);
        self.buffers[1] = buffer(device, "pixel triangles", &g.bvh.triangles, STORAGE);
        self.buffers[2] = buffer(device, "pixel surfaces", &g.surfaces, STORAGE);
        self.buffers[6] = buffer(
            device,
            "pixel objects",
            &g.objects(&scene.surfaces),
            STORAGE,
        );
        self.geometry = g;
        self.rebind(device);
    }

    /// New materials alone: the object table.
    pub fn set_surfaces(&mut self, device: &wgpu::Device, scene: &Scene) {
        let objects = self.geometry.objects(&scene.surfaces);
        self.buffers[6] = buffer(device, "pixel objects", &objects, STORAGE);
        self.rebind(device);
    }

    /// New volumes: their BVH and records.
    pub fn set_volumes(&mut self, device: &wgpu::Device, volumes: &[Volume]) {
        let media = Media::build(volumes);
        self.buffers[3] = buffer(device, "pixel medium nodes", &media.bvh.nodes, STORAGE);
        self.buffers[4] = buffer(
            device,
            "pixel medium triangles",
            &media.bvh.triangles,
            STORAGE,
        );
        self.buffers[5] = buffer(device, "pixel media", &media.records, STORAGE);
        self.volumes = media.counts;
        self.rebind(device);
    }

    /// New lights.
    pub fn set_lights(&mut self, device: &wgpu::Device, lights: &[Light]) {
        let packed: Vec<GpuLight> = lights.iter().map(GpuLight::of).collect();
        self.buffers[7] = buffer(device, "pixel lights", &packed, STORAGE);
        self.lights = packed.len() as u32;
        self.rebind(device);
    }

    fn rebind(&mut self, device: &wgpu::Device) {
        let all: Vec<_> = self
            .buffers
            .iter()
            .map(wgpu::Buffer::as_entire_binding)
            .collect();
        self.group = bind_group(device, "pixel scene", &self.layout, &all);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(z: f32, surface: u32) -> Mesh {
        let [a, b, c, d] = [[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]];
        Mesh {
            corners: vec![a, b, c, a, c, d],
            normals: None,
            tints: Some(vec![[0.5; 3]; 6]),
            surface,
            id: 7,
        }
    }

    #[test]
    fn meshes_flatten_with_their_objects_and_surfaces() {
        let g = Geometry::build(&[quad(0.0, 0), quad(1.0, 3)]);
        assert_eq!(g.vertices.len(), 12);
        assert_eq!(g.objects, [(0, 7), (3, 7)]);
        // Flat normals face +Z; each triangle's mean tint is its corners'.
        assert_eq!(g.vertices[0].normal, [0.0, 0.0, 1.0]);
        assert!(g.surfaces.iter().all(|s| s.tint == [0.5; 3]));
        assert_eq!(g.bounds(), Some(([0.0; 3], [1.0, 1.0, 1.0])));
        let surfaces = [crate::render::Surface {
            color: [0.2, 0.3, 0.4],
            translucency: 0.5,
            translucency_color: [1.0, 0.5, 0.0],
        }];
        let table = g.objects(&surfaces);
        assert_eq!(table[0].albedo, [0.2, 0.3, 0.4, 0.0]);
        assert_eq!(table[0].transmit, [0.5, 0.25, 0.0, 0.0]);
        // A material past the table is the default surface.
        assert_eq!(table[1].albedo, [0.8, 0.8, 0.8, 0.0]);
    }

    #[test]
    fn media_carry_their_volumes_index() {
        let box_of = |shadows| Volume {
            corners: quad(0.0, 0).corners,
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
