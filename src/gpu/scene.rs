//! The scene on the GPU, in two levels: the meshes in their own space (the
//! G-buffer's vertices and triangles, and each mesh's BVH) and the instances
//! placing them (their records and the top-level BVH over them). The scene
//! group every pass binds as group 1: the meshes' nodes and triangles, the
//! top level's nodes, the volumes' BVH (nodes, triangles, records), the
//! instances and the lights; beside it, the atlas of the pictures surfaces
//! sample, which the G-buffer reads. Each part is replaced only when its
//! source changes: a camera move touches none of it, an object's move the
//! instances alone. What changed is made and checked against the device
//! ([`Budget`]) before any of it replaces what the GPU holds.
use super::Changed;
use super::atlas::Atlas;
use super::budget::{Budget, Part};
use super::kernel::{bind_group, storage};
use super::records::{GpuInstance, GpuLight, GpuMedium, Vertex};
use crate::render::bvh::{Bvh, Node, Tlas, Triangle};
use crate::render::vector::V3;
use crate::render::{Instance, Mesh, Picture, Scene, Surface, Volume};
use kinogaki_render_api::display::wgpu;
use std::ops::Range;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// The most volumes a frame renders (a ray's inside set is a 32-bit mask).
pub const MAX_VOLUMES: usize = 32;

/// Where a mesh lies in the meshes' buffers.
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    /// Its triangles' run of indices.
    pub indices: Range<u32>,
    /// Its first vertex, which its indices count from.
    pub base: u32,
    /// Its BVH's root node.
    pub root: u32,
}

/// The meshes side by side: their vertices and triangles, and their BVHs in
/// one array of nodes and one of triangles, each mesh's indices moved past
/// the meshes before it.
pub struct Meshes {
    /// Each mesh's vertices, in a run.
    pub vertices: Vec<Vertex>,
    /// Three vertices a triangle, each mesh's counted from its first.
    pub indices: Vec<u32>,
    /// Where each mesh lies.
    pub meshes: Vec<Span>,
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
            indices: Vec::new(),
            meshes: Vec::new(),
            nodes: Vec::new(),
            triangles: Vec::new(),
        };
        for mesh in meshes {
            let tint = |v: u32| mesh.tints.as_ref().map_or([1.0; 4], |c| c[v as usize]);
            let base = made.vertices.len() as u32;
            made.vertices
                .extend((0..mesh.positions.len()).map(|v| Vertex {
                    position: mesh.positions[v],
                    normal: mesh.normals[v],
                    tint: tint(v as u32),
                    uv: mesh.uvs.as_ref().map_or([0.0; 2], |c| c[v]),
                }));
            let first = made.indices.len() as u32;
            made.indices.extend(mesh.triangles.iter().flatten());
            let (nodes, triangles) = (made.nodes.len() as u32, made.triangles.len() as u32);
            made.meshes.push(Span {
                indices: first..made.indices.len() as u32,
                base,
                root: nodes,
            });
            made.nodes.extend(mesh.bvh.nodes.iter().map(|n| Node {
                a: n.a + if n.count == 0 { nodes } else { triangles },
                ..*n
            }));
            let tinted = (mesh.bvh.triangles.iter())
                .zip(&mesh.bvh.order)
                .map(|(tri, &t)| {
                    let corners = mesh.triangles[t as usize].map(tint);
                    let [r, g, b] =
                        std::array::from_fn(|k| corners.iter().map(|c| c[k]).sum::<f32>() / 3.0);
                    let mut tri = *tri;
                    (tri.v0[3], tri.e1[3], tri.e2[3]) = (r, g, b);
                    tri
                });
            made.triangles.extend(tinted);
        }
        made
    }

    /// What they put on the device.
    fn parts(&self) -> [Part; 4] {
        [
            Part::of("its vertices", &self.vertices, false),
            Part::of("its vertex indices", &self.indices, false),
            Part::of("its hierarchy's nodes", &self.nodes, true),
            Part::of("its traced triangles", &self.triangles, true),
        ]
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
    /// A draw a mesh: where it lies, and its run of `drawn`.
    pub draws: Vec<(Span, Range<u32>)>,
}

impl Instances {
    /// `instances` of `meshes` (as [`Meshes::build`] laid out `bvhs`), with
    /// the materials `surfaces`, their pictures in `atlas`.
    pub fn build(
        instances: &[Instance],
        (meshes, bvhs): (&Meshes, &[Arc<Mesh>]),
        surfaces: &[Surface],
        atlas: &Atlas,
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
                    meshes.meshes[i.mesh as usize].root,
                    surface(surfaces, i.surface),
                    atlas,
                )
            })
            .collect();
        let mut drawn: Vec<u32> = (0..placed.len() as u32).collect();
        drawn.sort_by_key(|&k| placed[k as usize].mesh);
        let mut draws: Vec<(Span, Range<u32>)> = Vec::new();
        for (at, &k) in drawn.iter().enumerate() {
            let span = &meshes.meshes[placed[k as usize].mesh as usize];
            let at = at as u32;
            match draws.last_mut() {
                Some((s, run)) if s == span => run.end = at + 1,
                _ => draws.push((span.clone(), at..at + 1)),
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

    /// The records with the materials `surfaces`, their pictures in `atlas`.
    pub fn restyle(&mut self, surfaces: &[Surface], atlas: &Atlas) {
        for (record, &index) in self.records.iter_mut().zip(&self.surfaces) {
            record.set_surface(surface(surfaces, index), atlas);
        }
    }

    /// What they put on the device.
    fn parts(&self) -> [Part; 3] {
        [
            Part::of("its instances", &self.records, true),
            Part::of("its drawn instances", &self.drawn, false),
            Part::of("its instances' nodes", &self.tlas.nodes, true),
        ]
    }
}

/// Whether `a` and `b` are the same pictures, one for one.
fn same_pictures(a: &[Arc<Picture>], b: &[Arc<Picture>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| Arc::ptr_eq(a, b))
}

/// `atlas` as a texture of half-float texels.
fn texels(device: &wgpu::Device, queue: &wgpu::Queue, atlas: &Atlas) -> wgpu::TextureView {
    let [width, height] = atlas.size;
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("pixel atlas"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(&atlas.texels),
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
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

    /// What they put on the device.
    fn parts(&self) -> [Part; 3] {
        [
            Part::of("its volumes", &self.records, true),
            Part::of("its volumes' nodes", &self.bvh.nodes, true),
            Part::of("its volumes' triangles", &self.bvh.triangles, true),
        ]
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
const INDEX: wgpu::BufferUsages = wgpu::BufferUsages::INDEX;

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
    /// The meshes' vertices.
    pub vertices: wgpu::Buffer,
    /// Their triangles: three vertices each.
    pub indices: wgpu::Buffer,
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
    /// The pictures packed, as the atlas holds them.
    atlas: (Vec<Arc<Picture>>, Atlas),
    /// The atlas the G-buffer reads texels from.
    pub texels: wgpu::TextureView,
    /// Volumes rendered, and how many cast shadows.
    pub volumes: (u32, u32),
    /// Lights in the light buffer.
    pub lights: u32,
}

impl GpuScene {
    /// The buffers of an empty scene.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> GpuScene {
        let stages = wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT;
        let entries: Vec<_> = (0..8).map(|b| storage(b, stages)).collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pixel scene"),
            entries: &entries,
        });
        let buffers: [wgpu::Buffer; 8] =
            std::array::from_fn(|_| buffer::<[u32; 4]>(device, "pixel empty", &[], STORAGE));
        let meshes = Meshes::build(&[]);
        let atlas = Atlas::pack(&[]);
        let instances = Instances::build(&[], (&meshes, &[]), &[], &atlas);
        let mut made = GpuScene {
            texels: texels(device, queue, &atlas),
            atlas: (Vec::new(), atlas),
            vertices: buffer::<Vertex>(device, "pixel vertices", &[], VERTEX),
            indices: buffer::<u32>(device, "pixel indices", &[], INDEX),
            drawn: buffer::<u32>(device, "pixel drawn", &[], VERTEX),
            group: Self::bind(device, &layout, &buffers),
            layout,
            buffers,
            meshes,
            instances,
            volumes: (0, 0),
            lights: 0,
        };
        (made.set((device, queue), &Scene::default(), Changed::ALL))
            .expect("an empty scene fits any device");
        made
    }

    /// Takes what `changed` says of `scene`; nothing, saying why, when a
    /// part of it is larger than the device's buffers hold.
    pub fn set(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        scene: &Scene,
        changed: Changed,
    ) -> Result<(), String> {
        let budget = Budget::of(device);
        let repacked = changed.surfaces && !same_pictures(&self.atlas.0, &scene.pictures);
        let atlas = repacked.then(|| Atlas::pack(&scene.pictures));
        let meshes = changed.meshes.then(|| Meshes::build(&scene.meshes));
        let instances = (changed.meshes || changed.instances).then(|| {
            let laid = meshes.as_ref().unwrap_or(&self.meshes);
            let atlas = atlas.as_ref().unwrap_or(&self.atlas.1);
            Instances::build(
                &scene.instances,
                (laid, &scene.meshes),
                &scene.surfaces,
                atlas,
            )
        });
        let media = changed.volumes.then(|| Media::build(&scene.volumes));
        let lights = changed
            .lights
            .then(|| scene.lights.iter().map(GpuLight::of).collect::<Vec<_>>());
        let parts: Vec<Part> = (meshes.iter().flat_map(Meshes::parts))
            .chain(instances.iter().flat_map(Instances::parts))
            .chain(media.iter().flat_map(Media::parts))
            .chain(lights.iter().map(|l| Part::of("its lights", l, true)))
            .collect();
        budget.holds(&parts)?;
        // Everything fits: it replaces what the GPU holds.
        if let Some(atlas) = atlas {
            self.texels = texels(device, queue, &atlas);
            self.atlas = (scene.pictures.clone(), atlas);
        }
        if let Some(m) = meshes {
            self.vertices = buffer(device, "pixel vertices", &m.vertices, VERTEX);
            self.indices = buffer(device, "pixel indices", &m.indices, INDEX);
            self.buffers[NODES] = buffer(device, "pixel nodes", &m.nodes, STORAGE);
            self.buffers[TRIANGLES] = buffer(device, "pixel triangles", &m.triangles, STORAGE);
            self.meshes = m;
        }
        if let Some(i) = instances {
            self.drawn = buffer(device, "pixel drawn", &i.drawn, VERTEX);
            self.buffers[TOP] = buffer(device, "pixel top nodes", &i.tlas.nodes, STORAGE);
            self.buffers[INSTANCES] = buffer(device, "pixel instances", &i.records, STORAGE);
            self.instances = i;
        } else if changed.surfaces {
            self.instances.restyle(&scene.surfaces, &self.atlas.1);
            let records = &self.instances.records;
            self.buffers[INSTANCES] = buffer(device, "pixel instances", records, STORAGE);
        }
        if let Some(media) = media {
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
        if let Some(lights) = lights {
            self.buffers[LIGHTS] = buffer(device, "pixel lights", &lights, STORAGE);
            self.lights = lights.len() as u32;
        }
        if changed.any() {
            self.group = Self::bind(device, &self.layout, &self.buffers);
        }
        Ok(())
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

    /// A unit square at height `z`: its points and two triangles.
    fn square(z: f32) -> ([[f32; 3]; 4], [[u32; 3]; 2]) {
        let points = [[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]];
        (points, [[0, 1, 2], [0, 2, 3]])
    }

    fn quad(z: f32) -> Mesh {
        let (points, triangles) = square(z);
        let tints = Some(vec![[0.5; 4]; 6]);
        let held = crate::render::Corners {
            tints,
            ..Default::default()
        };
        Mesh::new(&points, &triangles, held)
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
            tint: None,
        }
    }

    #[test]
    fn meshes_lie_side_by_side_and_instances_draw_by_mesh() {
        let meshes = [Arc::new(quad(0.0)), Arc::new(quad(1.0))];
        let m = Meshes::build(&meshes);
        // A vertex a point (the diagonal's shared), three indices a triangle.
        assert_eq!((m.vertices.len(), m.indices.len()), (8, 12));
        assert_eq!(&m.indices[6..], [0, 1, 2, 0, 2, 3]);
        // Flat normals face +Z; each triangle's mean tint rides in its w's.
        assert_eq!(m.vertices[0].normal, [0.0, 0.0, 1.0]);
        assert!(
            m.triangles
                .iter()
                .all(|t| [t.v0[3], t.e1[3], t.e2[3]] == [0.5; 3])
        );
        // The second mesh's nodes and triangles are moved past the first's.
        let (first, second) = (&m.meshes[0], &m.meshes[1]);
        assert_eq!((first.indices.clone(), first.base), (0..6, 0));
        assert_eq!((second.indices.clone(), second.base), (6..12, 4));
        let root = m.nodes[second.root as usize];
        assert!(root.count > 0 && root.a >= meshes[0].bvh.triangles.len() as u32);
        let surfaces = [
            crate::render::Surface {
                color: [0.2, 0.3, 0.4],
                translucency: 0.5,
                translucency_color: [1.0, 0.5, 0.0],
                unlit: false,
                texture: None,
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
        let i = Instances::build(&placed, (&m, &meshes), &surfaces, &Atlas::pack(&[]));
        // Two draws: mesh 0 once, mesh 1 twice.
        let runs: Vec<_> = (i.draws.iter())
            .map(|(span, r)| (span.indices.clone(), r.len()))
            .collect();
        assert_eq!(runs, [(0..6, 1), (6..12, 2)]);
        assert_eq!(i.tlas.bounds(), Some(([0.0, 0.0, 0.0], [5.0, 1.0, 1.0])));
        let of = |dx: f32| i.records.iter().find(|r| r.to_world[0][3] == dx).unwrap();
        assert_eq!(of(0.0).albedo, [0.2, 0.3, 0.4, 0.0]);
        assert_eq!(of(0.0).transmit, [0.5, 0.25, 0.0, 0.0]);
        assert_eq!(of(0.0).id, [7, second.root, 0, 0]);
        // An unlit one says so in its transmittance's w; a material past the
        // table is the default surface.
        assert_eq!(of(4.0).transmit, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(of(2.0).albedo, [0.8, 0.8, 0.8, 0.0]);
    }

    #[test]
    fn media_carry_their_volumes_index() {
        let (points, triangles) = square(0.0);
        let corners: Vec<[f32; 3]> = (triangles.iter().flatten())
            .map(|&p| points[p as usize])
            .collect();
        let box_of = |shadows| Volume {
            corners: corners.clone(),
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
