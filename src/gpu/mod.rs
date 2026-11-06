//! The renderer on a host's wgpu device: the scene's buffers in two levels
//! (`scene`: the meshes and the instances placing them) and seven passes, recorded for one view at a time into a display image:
//!
//! 1. **G-buffer** (raster): albedo (the material's color × its tint),
//!    normal, depth, object id and the material's transmittance.
//! 2. **Ambient**: the domes' light × ray-traced AO.
//! 3. **Direct**: the lights with hard shadow rays, gobos and light through
//!    thin translucent surfaces, darkened by AO as AO on Direct asks.
//! 4. **Bounce**: one diffuse bounce, a fixed pattern of rays a pixel.
//! 5. **Volume**: each camera ray marched through the volumes and lit fog.
//! 6. **Shade**: the light (and rim) quantized and dithered through the
//!    Shade Ramp × albedo, the volumes over it, flat fog.
//! 7. **Upscale**: each film pixel a pixel-size block of the view.
//!
//! A [`Viewport`] is one view's uniform, film textures and bind groups.
use kinogaki_render_api::display::wgpu;
mod kernel;
mod passes;
pub mod readback;
mod records;
mod scene;
pub use scene::MAX_VOLUMES;
mod targets;

pub use records::Lighting;
pub use targets::{COLOR, Image, Target, Targets};

use crate::render::Scene;
use crate::render::camera::{Camera, CameraFrame};
use crate::render::film::Film;
use crate::render::settings::Settings;
use passes::{Groups, Passes};
use records::Frame;
use scene::GpuScene;

/// What changed in a scene since the renderer last took it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Changed {
    /// Meshes: their vertices and BVHs are laid out again (and the
    /// instances with them).
    pub meshes: bool,
    /// Instances: their records and the top level over them.
    pub instances: bool,
    /// Materials: the instances' records.
    pub surfaces: bool,
    /// Volumes: their small BVH and records.
    pub volumes: bool,
    /// Lights.
    pub lights: bool,
}

impl Changed {
    /// Everything.
    pub const ALL: Changed = Changed {
        meshes: true,
        instances: true,
        surfaces: true,
        volumes: true,
        lights: true,
    };

    /// Whether anything did.
    pub fn any(self) -> bool {
        self.meshes || self.instances || self.surfaces || self.volumes || self.lights
    }
}

/// The scene and passes on a host's device.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    passes: Passes,
    scene: GpuScene,
    /// The domes, rims and fog, read into each frame's uniform.
    ambient: crate::render::Ambient,
    rims: crate::render::Rims,
    fog: Option<crate::render::Fog>,
    /// Counts scene group replacements: a view bound to an older one binds
    /// again.
    generation: u64,
}

/// One view's frame: its uniform, its film textures and its bind groups.
pub struct Viewport {
    frame: wgpu::Buffer,
    targets: Option<Targets>,
    groups: Option<(u64, Groups)>,
}

impl Renderer {
    /// A renderer of an empty scene on `device` and `queue` (the host's own;
    /// wgpu handles are shared, not copied).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Renderer {
        let scene = GpuScene::new(device);
        Renderer {
            passes: Passes::new(device, &scene.layout),
            device: device.clone(),
            queue: queue.clone(),
            scene,
            ambient: Default::default(),
            rims: Default::default(),
            fog: None,
            generation: 0,
        }
    }

    /// The device it renders on.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue it submits to.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Takes `scene`, uploading what `changed` says.
    pub fn set_scene(&mut self, scene: &Scene, changed: Changed) {
        self.scene.set(&self.device, scene, changed);
        self.generation += u64::from(changed.any());
        (self.ambient, self.rims, self.fog) = (scene.ambient, scene.rims, scene.fog);
    }

    /// A new view's frame.
    pub fn viewport(&self) -> Viewport {
        Viewport {
            frame: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pixel frame"),
                size: std::mem::size_of::<Frame>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            targets: None,
            groups: None,
        }
    }

    /// Records a frame of `camera` at view size `view` with `settings` into
    /// `viewport` and display image `image` (its film copies too), into
    /// `encoder`; the frame's uniform is written to the queue ahead of it.
    /// None for an empty view.
    pub fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &mut Viewport,
        (camera, view, settings): (&Camera, [u32; 2], &Settings),
        image: &Image,
    ) -> Option<Film> {
        let film = Film::new(view[0], view[1], settings.pixel_size);
        if film.is_empty() {
            return None;
        }
        if viewport.targets.as_ref().is_none_or(|t| t.film != film) {
            viewport.targets = Some(Targets::new(&self.device, film));
            viewport.groups = None;
        }
        let targets = viewport.targets.as_ref()?;
        let frame = CameraFrame::new(camera, &film, self.scene.bounds());
        let lighting = Lighting {
            ambient: self.ambient,
            rims: self.rims,
            fog: self.fog,
            lights: self.scene.lights,
            volumes: self.scene.volumes,
            extent: self.scene.extent(),
        };
        let uniform = Frame::new(&frame, view, settings, &lighting);
        (self.queue).write_buffer(&viewport.frame, 0, bytemuck::bytes_of(&uniform));
        if viewport
            .groups
            .as_ref()
            .is_none_or(|(g, _)| *g != self.generation)
        {
            let groups = self.passes.bind(&self.device, &viewport.frame, targets);
            viewport.groups = Some((self.generation, groups));
        }
        let (_, groups) = viewport.groups.as_ref()?;
        let upscale = self
            .passes
            .bind_upscale(&self.device, &viewport.frame, targets, image);
        self.passes
            .encode(encoder, targets, &self.scene, groups, &upscale);
        image.copy(encoder, targets);
        Some(film)
    }

    /// The film textures `viewport`'s last frame wrote.
    pub fn targets<'a>(&self, viewport: &'a Viewport) -> Option<&'a Targets> {
        viewport.targets.as_ref()
    }
}
