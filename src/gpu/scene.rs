//! The scene on the GPU, in two levels: the meshes in their own space (the
//! G-buffer's vertices and triangles, and each mesh's BVH) and the instances
//! placing them (their records and the top-level BVH over them). The scene
//! group the G-buffer and the passes tracing rays bind as group 1: the
//! meshes' nodes and triangles, the top level's nodes, the volumes' BVH
//! (nodes, triangles, records), the instances, the lights, the cut-out
//! meshes' cut records and the atlas of the pictures surfaces sample. Each
//! part is replaced only when its source changes: a camera move touches none
//! of it, an object's move the instances alone, a deform its mesh's runs of
//! the meshes' buffers alone ([`Meshes`]). What changed is made and checked
//! against the device ([`Budget`]) before any of it replaces what the GPU
//! holds.
use super::Changed;
use super::atlas::Atlas;
use super::budget::{Budget, Part};
use super::kernel::{bind_group, storage, uniform};
use super::lights::GpuLight;
use super::meshes::{Meshes, Span};
use super::records::{GpuInstance, GpuMedium, Vertex};
use crate::render::bvh::{Bvh, Tlas};
use crate::render::vector::V3;
use crate::render::{Instance, Mesh, Picture, Scene, Surface, Volume, glow_meshes};
use kinogaki_render_api::display::wgpu;
use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// The most volumes a frame renders (a ray's inside set is a 32-bit mask).
pub const MAX_VOLUMES: usize = 32;

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
    /// the materials `surfaces`, their pictures in `atlas`: the first
    /// `traced` traced and drawn, the rest (the lights' shapes) drawn alone,
    /// after them, in no top-level leaf.
    pub fn build(
        (instances, traced): (&[Instance], usize),
        (meshes, bvhs): (&Meshes, &[Arc<Mesh>]),
        surfaces: &[Surface],
        atlas: &Atlas,
    ) -> Instances {
        let boxes: Vec<_> = (instances[..traced].iter())
            .map(|i| world_bounds(&i.to_world, bvhs[i.mesh as usize].bvh.bounds()))
            .collect();
        let tlas = Tlas::build(&boxes);
        let placed: Vec<&Instance> = (tlas.order.iter())
            .map(|&k| &instances[k as usize])
            .chain(&instances[traced..])
            .collect();
        let records = (placed.iter())
            .map(|i| {
                GpuInstance::of(
                    i,
                    &meshes.meshes[i.mesh as usize],
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

/// What the G-buffer draws: the scene's meshes, instances and materials,
/// then the unit shapes the lights' glows place, the glows (drawn alone,
/// traced by no ray) and their unlit materials.
struct Drawn<'a> {
    meshes: Cow<'a, [Arc<Mesh>]>,
    instances: Cow<'a, [Instance]>,
    surfaces: Cow<'a, [Surface]>,
}

impl Drawn<'_> {
    fn of(scene: &Scene) -> Drawn<'_> {
        if scene.glows.is_empty() {
            return Drawn {
                meshes: Cow::Borrowed(&scene.meshes),
                instances: Cow::Borrowed(&scene.instances),
                surfaces: Cow::Borrowed(&scene.surfaces),
            };
        }
        let mut meshes = scene.meshes.clone();
        let first = meshes.len() as u32;
        meshes.extend(glow_meshes().iter().cloned());
        let mut surfaces = scene.surfaces.clone();
        let mut instances = scene.instances.clone();
        for glow in &scene.glows {
            surfaces.push(Surface {
                color: glow.radiance,
                unlit: true,
                ..Surface::default()
            });
            instances.push(Instance {
                mesh: first + glow.mesh as u32,
                to_world: glow.to_world,
                to_own: None,
                surface: surfaces.len() as u32 - 1,
                id: 0,
                tint: None,
                alpha: None,
            });
        }
        Drawn {
            meshes: Cow::Owned(meshes),
            instances: Cow::Owned(instances),
            surfaces: Cow::Owned(surfaces),
        }
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
    /// One record a volume, then zeroed ones: [`MAX_VOLUMES`] records, the
    /// uniform the passes read.
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
        records.resize(MAX_VOLUMES, GpuMedium::default());
        Media {
            records,
            bvh,
            counts,
        }
    }

    /// What they put on the device beside their uniform.
    fn parts(&self) -> [Part; 2] {
        [
            Part::of("its volumes' nodes", &self.bvh.nodes, true),
            Part::of("its volumes' triangles", &self.bvh.triangles, true),
        ]
    }
}

/// `runs` of `items` written over the same places of `buffer`.
fn write<T: bytemuck::Pod>(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    items: &[T],
    runs: &[Range<usize>],
) {
    let size = std::mem::size_of::<T>();
    for run in runs.iter().filter(|r| !r.is_empty()) {
        let bytes = bytemuck::cast_slice(&items[run.clone()]);
        queue.write_buffer(buffer, (run.start * size) as u64, bytes);
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

const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(COPY);
const UNIFORM: wgpu::BufferUsages = wgpu::BufferUsages::UNIFORM.union(COPY);
const VERTEX: wgpu::BufferUsages = wgpu::BufferUsages::VERTEX.union(COPY);
const INDEX: wgpu::BufferUsages = wgpu::BufferUsages::INDEX.union(COPY);
/// Every buffer takes writes in place.
const COPY: wgpu::BufferUsages = wgpu::BufferUsages::COPY_DST;

/// The scene group's bindings: its buffers (storage, but the volumes'
/// records: a uniform, which leaves the passes within wgpu's default 8
/// storage buffers), then the atlas.
const NODES: usize = 0;
const TRIANGLES: usize = 1;
const TOP: usize = 2;
const MEDIUM_NODES: usize = 3;
const MEDIUM_TRIANGLES: usize = 4;
const MEDIA: usize = 5;
const INSTANCES: usize = 6;
const LIGHTS: usize = 7;
const CUTS: usize = 8;
const BUFFERS: usize = 9;

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
    buffers: [wgpu::Buffer; BUFFERS],
    /// The meshes the buffers hold.
    meshes: Meshes,
    /// The instances the buffers hold.
    pub instances: Instances,
    /// The pictures packed, as the atlas holds them.
    atlas: (Vec<Arc<Picture>>, Atlas),
    /// The atlas surfaces read texels from.
    texels: wgpu::TextureView,
    /// Volumes rendered, and how many cast shadows.
    pub volumes: (u32, u32),
    /// Lights in the light buffer.
    pub lights: u32,
    /// How many times the meshes were laid out whole (tests and counters).
    pub layouts: u64,
}

impl GpuScene {
    /// The buffers of an empty scene.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> GpuScene {
        let stages = wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT;
        let atlas = wgpu::BindGroupLayoutEntry {
            binding: BUFFERS as u32,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let entries: Vec<_> = (0..BUFFERS)
            .map(|b| match b {
                MEDIA => uniform(b as u32, stages),
                _ => storage(b as u32, stages),
            })
            .chain([atlas])
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pixel scene"),
            entries: &entries,
        });
        let buffers: [wgpu::Buffer; BUFFERS] = std::array::from_fn(|b| {
            let usage = if b == MEDIA { UNIFORM } else { STORAGE };
            buffer::<[u32; 4]>(device, "pixel empty", &[], usage)
        });
        let meshes = Meshes::build(&[]);
        let atlas = Atlas::pack(&[]);
        let instances = Instances::build((&[], 0), (&meshes, &[]), &[], &atlas);
        let texels = texels(device, queue, &atlas);
        let mut made = GpuScene {
            group: Self::bind(device, &layout, &buffers, &texels),
            texels,
            atlas: (Vec::new(), atlas),
            vertices: buffer::<Vertex>(device, "pixel vertices", &[], VERTEX),
            indices: buffer::<u32>(device, "pixel indices", &[], INDEX),
            drawn: buffer::<u32>(device, "pixel drawn", &[], VERTEX),
            layout,
            buffers,
            meshes,
            instances,
            volumes: (0, 0),
            lights: 0,
            layouts: 0,
        };
        made.lay_out(device);
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
        let drawn = Drawn::of(scene);
        let written = (changed.meshes).then(|| self.meshes.rewrite(&drawn.meshes));
        let meshes = matches!(written, Some(None)).then(|| Meshes::build(&drawn.meshes));
        let instances = (changed.meshes || changed.instances).then(|| {
            let laid = meshes.as_ref().unwrap_or(&self.meshes);
            let atlas = atlas.as_ref().unwrap_or(&self.atlas.1);
            Instances::build(
                (&drawn.instances, scene.instances.len()),
                (laid, &drawn.meshes),
                &drawn.surfaces,
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
        if let Err(why) = budget.holds(&parts) {
            // What was written in place never reaches the GPU: the next
            // scene is laid out whole.
            self.meshes.unsent();
            return Err(why);
        }
        // Everything fits: it replaces what the GPU holds.
        if let Some(atlas) = atlas {
            self.texels = texels(device, queue, &atlas);
            self.atlas = (scene.pictures.clone(), atlas);
        }
        if let Some(m) = meshes {
            self.meshes = m;
            self.lay_out(device);
            self.layouts += 1;
        } else if let Some(Some(written)) = written {
            let [vertices, indices, nodes, triangles, cuts] = &written;
            let m = &self.meshes;
            write(queue, &self.vertices, &m.vertices, vertices);
            write(queue, &self.indices, &m.indices, indices);
            write(queue, &self.buffers[NODES], &m.nodes, nodes);
            write(queue, &self.buffers[TRIANGLES], &m.triangles, triangles);
            write(queue, &self.buffers[CUTS], &m.cuts, cuts);
        }
        if let Some(i) = instances {
            self.drawn = buffer(device, "pixel drawn", &i.drawn, VERTEX);
            self.buffers[TOP] = buffer(device, "pixel top nodes", &i.tlas.nodes, STORAGE);
            self.buffers[INSTANCES] = buffer(device, "pixel instances", &i.records, STORAGE);
            self.instances = i;
        } else if changed.surfaces {
            self.instances.restyle(&drawn.surfaces, &self.atlas.1);
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
            self.buffers[MEDIA] = buffer(device, "pixel media", &media.records, UNIFORM);
            self.volumes = media.counts;
        }
        if let Some(lights) = lights {
            self.buffers[LIGHTS] = buffer(device, "pixel lights", &lights, STORAGE);
            self.lights = lights.len() as u32;
        }
        if changed.any() {
            self.group = Self::bind(device, &self.layout, &self.buffers, &self.texels);
        }
        Ok(())
    }

    /// The meshes' buffers made whole from the meshes.
    fn lay_out(&mut self, device: &wgpu::Device) {
        let m = &self.meshes;
        self.vertices = buffer(device, "pixel vertices", &m.vertices, VERTEX);
        self.indices = buffer(device, "pixel indices", &m.indices, INDEX);
        self.buffers[NODES] = buffer(device, "pixel nodes", &m.nodes, STORAGE);
        self.buffers[TRIANGLES] = buffer(device, "pixel triangles", &m.triangles, STORAGE);
        self.buffers[CUTS] = buffer(device, "pixel cuts", &m.cuts, STORAGE);
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
        buffers: &[wgpu::Buffer; BUFFERS],
        texels: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let all: Vec<_> = (buffers.iter())
            .map(wgpu::Buffer::as_entire_binding)
            .chain([wgpu::BindingResource::TextureView(texels)])
            .collect();
        bind_group(device, "pixel scene", layout, &all)
    }
}

#[cfg(test)]
mod tests {
    use super::super::records::{OWN_ALPHA, OWN_COLOR};
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
            alpha: None,
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
        let i = Instances::build(
            (&placed, placed.len()),
            (&m, &meshes),
            &surfaces,
            &Atlas::pack(&[]),
        );
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
        // An instance's own color and alpha, each said in `id.z`.
        let own = |tint, alpha| {
            let placed = [Instance {
                tint,
                alpha,
                ..instance(0, 0.0, 0)
            }];
            let i = Instances::build(
                (&placed, placed.len()),
                (&m, &meshes),
                &surfaces,
                &Atlas::pack(&[]),
            );
            (i.records[0].id[2], i.records[0].tint)
        };
        assert_eq!(own(None, None), (0, [0.0, 0.0, 0.0, 1.0]));
        assert_eq!(own(Some([0.5; 3]), None), (OWN_COLOR, [0.5, 0.5, 0.5, 1.0]));
        assert_eq!(own(None, Some(0.25)), (OWN_ALPHA, [0.0, 0.0, 0.0, 0.25]));
    }

    /// A cut-out mesh's traced triangles keep what the G-buffer reads at
    /// their corners, in their order, where its span says; another mesh
    /// keeps none.
    #[test]
    fn a_cut_out_meshs_traced_triangles_keep_their_corners() {
        let (points, triangles) = square(1.0);
        let held = crate::render::Corners {
            tints: Some((0..6).map(|c| [1.0, 1.0, 1.0, c as f32 / 10.0]).collect()),
            uvs: Some((0..6).map(|c| [c as f32, 10.0 + c as f32]).collect()),
            ..Default::default()
        };
        let cut = Mesh {
            cut_out: true,
            ..Mesh::new(&points, &triangles, held)
        };
        let m = Meshes::build(&[Arc::new(quad(0.0)), Arc::new(cut.clone())]);
        assert_eq!(m.cuts.len(), 2, "the second mesh's alone");
        let span = &m.meshes[1];
        assert_eq!((m.meshes[0].cuts, span.cuts), (0, 2));
        for (k, &t) in cut.bvh.order.iter().enumerate() {
            let record = m.cuts[k + 2 - span.cuts as usize];
            let corner = |c: usize| (t as usize * 3 + c) as f32;
            assert_eq!(record.s, [0, 1, 2].map(corner));
            assert_eq!(record.t, [0, 1, 2].map(|c| 10.0 + corner(c)));
            assert_eq!(record.alpha, [0, 1, 2].map(|c| corner(c) / 10.0));
        }
    }

    /// A mesh made again that fits its runs is written there, those runs
    /// alone said written; a mesh outgrowing them, or another count of
    /// meshes, is laid out again.
    #[test]
    fn a_mesh_made_again_is_written_where_it_lies() {
        let meshes = [Arc::new(quad(0.0)), Arc::new(quad(1.0))];
        let mut m = Meshes::build(&meshes);
        let spans = m.meshes.clone();
        let raised = [meshes[0].clone(), Arc::new(quad(2.0))];
        let written = m.rewrite(&raised).expect("it fits");
        assert_eq!(m.meshes, spans, "where it lies");
        let runs = |k: usize| {
            written[k]
                .iter()
                .map(|r| (r.start, r.end))
                .collect::<Vec<_>>()
        };
        assert_eq!(runs(0), [(4, 8)], "the second mesh's vertices alone");
        assert_eq!(runs(1), [(6, 12)]);
        assert_eq!(m.vertices[4].position, [0.0, 0.0, 2.0]);
        assert_eq!(m.vertices[0].position, [0.0, 0.0, 0.0], "the first kept");
        let again = m.rewrite(&raised).expect("the same meshes");
        assert!(again.iter().all(Vec::is_empty), "nothing written");
        let (points, _) = square(3.0);
        let larger = Mesh::new(
            &points,
            &[[0, 1, 2], [0, 2, 3], [1, 2, 3]],
            Default::default(),
        );
        assert!(m.rewrite(&[meshes[0].clone(), Arc::new(larger)]).is_none());
        assert!(m.rewrite(&meshes[..1]).is_none());
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
        assert_eq!(Media::build(&[]).records.len(), MAX_VOLUMES);
    }
}
