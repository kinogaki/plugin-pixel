//! The lighting passes, all ray traced against the BVH, in the order they
//! run: the domes' light × AO
//! (shaders/ambient.wgsl), direct light from the lights with hard shadows,
//! gobos and translucency (shadow rays keep what thin translucent surfaces
//! let through), darkened by AO as AO on Direct asks
//! (shaders/direct.wgsl), and one diffuse bounce (shaders/bounce.wgsl).

use super::kernel::Kernel;
use super::layout::{self, float_texture, storage_texture, uint_texture};
use super::scene::GpuScene;
use super::targets::{COLOR, SCALAR, Targets};
use super::timing::Timer;

/// AO rays a film pixel (`AO_RAYS` in ambient.wgsl): a fixed set, the same
/// for every pixel and frame.
pub const AO_RAYS: u32 = 32;

/// The WGSL the passes that light share: the volumes (whose shadows the
/// lights' shadow rays keep), the gobo, the lights and the hemisphere
/// pattern.
pub const MEDIUM: &str = include_str!("../shaders/medium.wgsl");
pub const GOBO: &str = include_str!("../shaders/gobo.wgsl");
pub const LIGHT: &str = include_str!("../shaders/light.wgsl");
pub const HEMISPHERE: &str = include_str!("../shaders/hemisphere.wgsl");

/// The ambient, direct and bounce kernels.
pub struct LightingPasses {
    ambient: Kernel,
    direct: Kernel,
    bounce: Kernel,
}

/// A view's lighting groups, as [`LightingPasses::encode`] takes them.
pub struct LightingGroups {
    ambient: wgpu::BindGroup,
    direct: wgpu::BindGroup,
    bounce: wgpu::BindGroup,
}

impl LightingPasses {
    /// The kernels; `bvh` is their group 1.
    pub fn new(device: &wgpu::Device, bvh: &wgpu::BindGroupLayout) -> LightingPasses {
        let c = wgpu::ShaderStages::COMPUTE;
        let bvh_source = layout::bvh_source();
        let rays = format!("const AO_RAYS: u32 = {AO_RAYS}u;");
        // What every pass reading the G-buffer binds first; the lights
        // (binding 4) where they are read.
        let gbuffer = [
            layout::uniform(0, c),
            float_texture(1),
            float_texture(2),
            uint_texture(3),
        ];
        let ambient = Kernel::new(
            device,
            "pixel ambient",
            &[
                &bvh_source,
                HEMISPHERE,
                &rays,
                include_str!("../shaders/ambient.wgsl"),
            ],
            &[
                &gbuffer[..],
                &[storage_texture(4, COLOR), storage_texture(5, SCALAR)],
            ]
            .concat(),
            &[bvh],
        );
        let direct = Kernel::new(
            device,
            "pixel direct",
            &[
                &bvh_source,
                MEDIUM,
                GOBO,
                LIGHT,
                include_str!("../shaders/direct.wgsl"),
            ],
            &[
                &gbuffer[..],
                &[
                    layout::storage(4, c),
                    float_texture(5),
                    float_texture(6),
                    storage_texture(7, COLOR),
                ],
            ]
            .concat(),
            &[bvh],
        );
        let bounce = Kernel::new(
            device,
            "pixel bounce",
            &[
                &bvh_source,
                MEDIUM,
                GOBO,
                LIGHT,
                HEMISPHERE,
                include_str!("../shaders/bounce.wgsl"),
            ],
            &[
                &gbuffer[..],
                &[layout::storage(4, c), storage_texture(5, COLOR)],
            ]
            .concat(),
            &[bvh],
        );
        LightingPasses {
            ambient,
            direct,
            bounce,
        }
    }

    /// A view's groups: its G-buffer, the lights and its outputs.
    pub fn bind<'a>(
        &self,
        device: &wgpu::Device,
        frame: &'a wgpu::Buffer,
        t: &'a Targets,
        scene: &'a GpuScene,
    ) -> LightingGroups {
        use wgpu::BindingResource::TextureView as Tex;
        let gbuffer = [
            frame.as_entire_binding(),
            Tex(&t.normal.view),
            Tex(&t.depth.view),
            Tex(&t.id.view),
        ];
        let with = |more: Vec<wgpu::BindingResource<'a>>| [gbuffer.to_vec(), more].concat();
        LightingGroups {
            ambient: self
                .ambient
                .bind(device, &with(vec![Tex(&t.ambient.view), Tex(&t.ao.view)])),
            direct: self.direct.bind(
                device,
                &with(vec![
                    scene.lights.as_entire_binding(),
                    Tex(&t.transmit.view),
                    Tex(&t.ao.view),
                    Tex(&t.direct.view),
                ]),
            ),
            bounce: self.bounce.bind(
                device,
                &with(vec![
                    scene.lights.as_entire_binding(),
                    Tex(&t.indirect.view),
                ]),
            ),
        }
    }

    /// Records the three passes over the film with a view's `groups`.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        t: &Targets,
        scene: &GpuScene,
        groups: &LightingGroups,
        timer: Option<&Timer>,
    ) {
        let size = t.film.size;
        let passes = [
            (&self.ambient, &groups.ambient),
            (&self.direct, &groups.direct),
            (&self.bounce, &groups.bounce),
        ];
        for (slot, (kernel, group)) in passes.into_iter().enumerate() {
            kernel.dispatch(encoder, size, &[group, &scene.bvh], timer, slot + 1);
        }
    }
}
