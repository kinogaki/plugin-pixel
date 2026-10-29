//! The scene on the GPU: vertices, the object table, the lights, the BVH
//! with its triangles' surfaces, and the volumes' BVH and records. Each
//! part is replaced only when its source changes.

use wgpu::util::DeviceExt;

use kinogaki_pixel_render::geometry::{Geometry, Object};
use kinogaki_pixel_render::lights::GpuLight;
use kinogaki_pixel_render::media::Media;

use super::layout;

/// The surface BVH's buffers (nodes, triangles, their surfaces) or the
/// volumes' (nodes, triangles, records).
struct Tree([wgpu::Buffer; 3]);

/// The scene's buffers and the BVHs' bind group (group 1 of the
/// ray-tracing passes: the surface BVH at bindings 0 to 2, the volumes' at
/// 3 to 5, the object table at 6: what shadow rays read a hit's
/// transmittance from).
pub struct GpuScene {
    /// G-buffer vertices, three a triangle.
    pub vertices: wgpu::Buffer,
    /// Vertices to draw.
    pub vertex_count: u32,
    /// The object table.
    pub objects: wgpu::Buffer,
    /// The lights but domes.
    pub lights: wgpu::Buffer,
    /// The BVH's layout.
    pub bvh_layout: wgpu::BindGroupLayout,
    /// Both BVHs, bound.
    pub bvh: wgpu::BindGroup,
    surfaces: Tree,
    media: Tree,
}

/// A storage buffer holding `data`, or one zeroed `T` when empty (bindings
/// may not be empty).
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

const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST);

impl GpuScene {
    /// The buffers of an empty scene.
    pub fn new(device: &wgpu::Device) -> GpuScene {
        let entries: Vec<_> = (0..7)
            .map(|b| layout::storage(b, wgpu::ShaderStages::COMPUTE))
            .collect();
        let bvh_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pixel bvh"),
            entries: &entries,
        });
        let empty = Geometry::build(&[]);
        let surfaces = Self::surface_tree(device, &empty);
        let media = Self::media_tree(device, &Media::build(&[]));
        let objects = buffer::<Object>(device, "pixel objects", &[], STORAGE);
        let bvh = Self::bind(device, &bvh_layout, &surfaces, &media, &objects);
        GpuScene {
            vertices: Self::vertices(device, &empty),
            vertex_count: 0,
            objects,
            lights: buffer::<GpuLight>(device, "pixel lights", &[], STORAGE),
            bvh_layout,
            bvh,
            surfaces,
            media,
        }
    }

    fn surface_tree(device: &wgpu::Device, geometry: &Geometry) -> Tree {
        Tree([
            buffer(device, "pixel bvh nodes", &geometry.bvh.nodes, STORAGE),
            buffer(
                device,
                "pixel bvh triangles",
                &geometry.bvh.triangles,
                STORAGE,
            ),
            buffer(device, "pixel surfaces", &geometry.surfaces, STORAGE),
        ])
    }

    fn media_tree(device: &wgpu::Device, media: &Media) -> Tree {
        Tree([
            buffer(device, "pixel media nodes", &media.bvh.nodes, STORAGE),
            buffer(
                device,
                "pixel media triangles",
                &media.bvh.triangles,
                STORAGE,
            ),
            buffer(device, "pixel media", &media.records, STORAGE),
        ])
    }

    fn vertices(device: &wgpu::Device, geometry: &Geometry) -> wgpu::Buffer {
        buffer(
            device,
            "pixel vertices",
            &geometry.vertices,
            wgpu::BufferUsages::VERTEX,
        )
    }

    fn bind(
        device: &wgpu::Device,
        bvh_layout: &wgpu::BindGroupLayout,
        surfaces: &Tree,
        media: &Tree,
        objects: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let all: Vec<_> = surfaces
            .0
            .iter()
            .chain(&media.0)
            .chain([objects])
            .map(wgpu::Buffer::as_entire_binding)
            .collect();
        layout::bind_group(device, "pixel bvh", bvh_layout, &all)
    }

    /// New geometry: vertices and the surface BVH.
    pub fn set_geometry(&mut self, device: &wgpu::Device, geometry: &Geometry) {
        self.surfaces = Self::surface_tree(device, geometry);
        self.vertices = Self::vertices(device, geometry);
        self.vertex_count = geometry.vertices.len() as u32;
        self.rebind(device);
    }

    /// New volumes: their BVH and records.
    pub fn set_media(&mut self, device: &wgpu::Device, media: &Media) {
        self.media = Self::media_tree(device, media);
        self.rebind(device);
    }

    fn rebind(&mut self, device: &wgpu::Device) {
        self.bvh = Self::bind(
            device,
            &self.bvh_layout,
            &self.surfaces,
            &self.media,
            &self.objects,
        );
    }

    /// A new object table. Returns whether the buffer was replaced (its
    /// bind groups must then be remade; the BVHs' is here).
    pub fn set_objects(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        objects: &[Object],
    ) -> bool {
        let replaced = Self::write(device, queue, &mut self.objects, "pixel objects", objects);
        if replaced {
            self.rebind(device);
        }
        replaced
    }

    /// New lights; as [`GpuScene::set_objects`].
    pub fn set_lights(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lights: &[GpuLight],
    ) -> bool {
        Self::write(device, queue, &mut self.lights, "pixel lights", lights)
    }

    /// Writes `data` in place when it fits, else replaces the buffer.
    fn write<T: bytemuck::Pod + Default>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &mut wgpu::Buffer,
        label: &str,
        data: &[T],
    ) -> bool {
        let bytes = std::mem::size_of_val(data) as u64;
        if bytes <= target.size() {
            if bytes > 0 {
                queue.write_buffer(target, 0, bytemuck::cast_slice(data));
            }
            return false;
        }
        *target = buffer(device, label, data, STORAGE);
        true
    }
}
