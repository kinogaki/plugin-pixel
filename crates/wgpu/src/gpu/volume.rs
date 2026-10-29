//! The volume pass: each film pixel's camera ray marched through the
//! volumes and lit fog up to the first surface (shaders/volume.wgsl), into
//! the volume output the shade pass composes.

use super::kernel::Kernel;
use super::layout::{self, float_texture, storage_texture, uint_texture};
use super::lighting::{GOBO, HEMISPHERE, LIGHT, MEDIUM};
use super::scene::GpuScene;
use super::targets::{COLOR, Targets};
use super::timing::Timer;

/// The volume kernel.
pub struct VolumePass {
    kernel: Kernel,
}

impl VolumePass {
    /// The kernel; `bvh` (both BVHs) is its group 1.
    pub fn new(device: &wgpu::Device, bvh: &wgpu::BindGroupLayout) -> VolumePass {
        let c = wgpu::ShaderStages::COMPUTE;
        let kernel = Kernel::new(
            device,
            "pixel volume",
            &[
                &layout::bvh_source(),
                MEDIUM,
                GOBO,
                LIGHT,
                HEMISPHERE,
                include_str!("../shaders/volume.wgsl"),
            ],
            &[
                layout::uniform(0, c),
                float_texture(1),
                uint_texture(2),
                storage_texture(3, COLOR),
                layout::storage(4, c),
            ],
            &[bvh],
        );
        VolumePass { kernel }
    }

    /// A view's group: its depth, ids, volume output and the lights.
    pub fn bind(
        &self,
        device: &wgpu::Device,
        frame: &wgpu::Buffer,
        t: &Targets,
        scene: &GpuScene,
    ) -> wgpu::BindGroup {
        use wgpu::BindingResource::TextureView as Tex;
        self.kernel.bind(
            device,
            &[
                frame.as_entire_binding(),
                Tex(&t.depth.view),
                Tex(&t.id.view),
                Tex(&t.volume.view),
                scene.lights.as_entire_binding(),
            ],
        )
    }

    /// Records the pass over the film.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        t: &Targets,
        scene: &GpuScene,
        group: &wgpu::BindGroup,
        timer: Option<&Timer>,
    ) {
        self.kernel
            .dispatch(encoder, t.film.size, &[group, &scene.bvh], timer, 4);
    }
}
