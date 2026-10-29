//! The shade pass (rim, quantize and dither through the Shade Ramp, ×
//! albedo, the volumes over it, fog: shaders/shade.wgsl) and the
//! upscale to the view (shaders/upscale.wgsl).

use super::kernel::Kernel;
use super::layout::{self, float_texture, storage_texture, uint_texture};
use super::targets::{COLOR, Target, Targets};
use super::timing::Timer;

/// The shade and upscale kernels.
pub struct ShadePasses {
    shade: Kernel,
    upscale: Kernel,
}

impl ShadePasses {
    /// The kernels.
    pub fn new(device: &wgpu::Device) -> ShadePasses {
        let c = wgpu::ShaderStages::COMPUTE;
        let shade = Kernel::new(
            device,
            "pixel shade",
            &[include_str!("../shaders/shade.wgsl")],
            &[
                layout::uniform(0, c),
                float_texture(1),
                float_texture(2),
                float_texture(3),
                float_texture(4),
                uint_texture(5),
                float_texture(6),
                float_texture(7),
                storage_texture(8, COLOR),
                float_texture(9),
            ],
            &[],
        );
        let upscale = Kernel::new(
            device,
            "pixel upscale",
            &[include_str!("../shaders/upscale.wgsl")],
            &[
                layout::uniform(0, c),
                float_texture(1),
                storage_texture(2, COLOR),
            ],
            &[],
        );
        ShadePasses { shade, upscale }
    }

    /// A view's shade group: its lighting, albedo, normal, depth, beauty and
    /// volumes.
    pub fn bind(
        &self,
        device: &wgpu::Device,
        frame: &wgpu::Buffer,
        t: &Targets,
    ) -> wgpu::BindGroup {
        use wgpu::BindingResource::TextureView as Tex;
        self.shade.bind(
            device,
            &[
                frame.as_entire_binding(),
                Tex(&t.albedo.view),
                Tex(&t.direct.view),
                Tex(&t.ambient.view),
                Tex(&t.indirect.view),
                Tex(&t.id.view),
                Tex(&t.normal.view),
                Tex(&t.depth.view),
                Tex(&t.beauty.view),
                Tex(&t.volume.view),
            ],
        )
    }

    /// A view's upscale group: its beauty into `image` (view size).
    pub fn bind_upscale(
        &self,
        device: &wgpu::Device,
        frame: &wgpu::Buffer,
        t: &Targets,
        image: &Target,
    ) -> wgpu::BindGroup {
        use wgpu::BindingResource::TextureView as Tex;
        self.upscale.bind(
            device,
            &[
                frame.as_entire_binding(),
                Tex(&t.beauty.view),
                Tex(&image.view),
            ],
        )
    }

    /// Records the shade pass over the film, then the upscale over the view.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        t: &Targets,
        shade: &wgpu::BindGroup,
        upscale: &wgpu::BindGroup,
        timer: Option<&Timer>,
    ) {
        self.shade
            .dispatch(encoder, t.film.size, &[shade], timer, 5);
        self.upscale
            .dispatch(encoder, t.film.view, &[upscale], timer, 6);
    }
}
