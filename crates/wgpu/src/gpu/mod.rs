//! The renderer on a host's wgpu device: the scene's buffers and the seven
//! passes (G-buffer, ambient, direct, indirect, volume, shade, upscale) in
//! [`Renderer`], and each view's own frame uniform, film textures, display
//! images (with their film-size companions, [`film`]) and bind groups in a
//! [`Viewport`], so several views render from one scene in one submission.

mod film;
mod frame;
mod gbuffer;
mod kernel;
mod layout;
mod lighting;
mod readback;
mod scene;
mod shade;
mod targets;
mod timing;
mod volume;

pub use lighting::AO_RAYS;
pub use readback::{f16_to_f32, read as read_texture};
pub use timing::{PASSES, PassTimes};

use kinogaki_pixel_render::camera::CameraFrame;
use kinogaki_pixel_render::geometry::Geometry;
use kinogaki_pixel_render::lights::{GpuLight, pack_light};
use kinogaki_pixel_render::media::Media;
use kinogaki_pixel_render::{
    Aov, AovImage, Camera, Dome, Film, Fog, Lighting, Mesh, PixelSurface, Rims, Scene, Settings,
    Volume,
};

use film::Companions;
use frame::{FrameInputs, FrameUniform};
use gbuffer::GbufferPass;
use lighting::{LightingGroups, LightingPasses};
use scene::GpuScene;
use shade::ShadePasses;
use targets::{Target, Targets};
use timing::Timer;
use volume::VolumePass;

/// What a frame looks through: the camera and the view's size in view
/// (screen) pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    /// The camera.
    pub camera: Camera,
    /// The view's width, in view pixels.
    pub width: u32,
    /// The view's height, in view pixels.
    pub height: u32,
}

/// Pixel Render's scene and passes on a host's device. Scene setters upload
/// only what they change: [`Renderer::set_meshes`] rebuilds the BVH and
/// [`Renderer::set_volumes`] the volumes' small one; materials, lights,
/// settings and cameras never do.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    settings: Settings,
    geometry: Geometry,
    media: Media,
    materials: Vec<PixelSurface>,
    lights: u32,
    dome: Option<Dome>,
    rims: Rims,
    fog: Option<Fog>,
    scene: GpuScene,
    gbuffer: GbufferPass,
    lighting: LightingPasses,
    volume: VolumePass,
    shade: ShadePasses,
    /// Counts scene buffer replacements: views bound to an older one bind
    /// again.
    generation: u64,
    geometry_builds: u64,
}

/// A view's bind groups, for one scene generation and its targets.
struct Groups {
    generation: u64,
    gbuffer: wgpu::BindGroup,
    lighting: LightingGroups,
    volume: wgpu::BindGroup,
    shade: wgpu::BindGroup,
    /// The upscale into each display image.
    upscale: Vec<wgpu::BindGroup>,
}

/// One view's frame: its uniform, its film textures, its display images
/// (the upscaled beauty; a host shows one while the next is written) with
/// their film-size companions, and its timestamps.
pub struct Viewport {
    device: wgpu::Device,
    queue: wgpu::Queue,
    frame: wgpu::Buffer,
    targets: Option<Targets>,
    images: Vec<Target>,
    /// Each display image's film-size copies of its frame's outputs.
    companions: Vec<Companions>,
    /// The outputs the companions keep beside the beauty.
    kept: Vec<Aov>,
    /// Which set of display images it holds: unique across viewports, new
    /// each time they are made (the film changed size).
    images_id: u64,
    count: usize,
    groups: Option<Groups>,
    timer: Option<Timer>,
}

/// Display image sets made so far (their ids).
static IMAGE_SETS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Viewport {
    /// A view of `images` display images (at least 1) on `renderer`'s
    /// device; its passes are timed when the device has `TIMESTAMP_QUERY`.
    pub fn new(renderer: &Renderer, images: usize) -> Viewport {
        let device = &renderer.device;
        Viewport {
            device: device.clone(),
            queue: renderer.queue.clone(),
            frame: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pixel frame"),
                size: std::mem::size_of::<FrameUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            targets: None,
            images: Vec::new(),
            companions: Vec::new(),
            kept: Vec::new(),
            images_id: 0,
            count: images.max(1),
            groups: None,
            timer: Timer::new(device, &renderer.queue),
        }
    }

    /// The last frame's film, if any.
    pub fn film(&self) -> Option<Film> {
        self.targets.as_ref().map(|t| t.film)
    }

    /// Which set of display images it holds (new when the film changes
    /// size; 0 before the first frame).
    pub fn images_id(&self) -> u64 {
        self.images_id
    }

    /// Display images it holds.
    pub fn images(&self) -> usize {
        self.images.len()
    }

    /// Display image `at`: the beauty, each film pixel repeated over its
    /// `pixel_size`² block (Rgba16Float, ACEScg).
    pub fn image(&self, at: usize) -> Option<&wgpu::Texture> {
        self.images.get(at).map(|t| &t.texture)
    }

    /// Keep `aovs` (beside the beauty) in the display images' film-size
    /// companions from the next frame on.
    pub fn keep(&mut self, aovs: &[Aov]) {
        self.kept = aovs.to_vec();
    }

    /// The outputs the companions keep beside the beauty.
    pub fn kept(&self) -> &[Aov] {
        &self.kept
    }

    /// Display image `at`'s film-size copy of `aov` (the beauty always,
    /// other outputs once kept and written).
    pub fn film_image(&self, at: usize, aov: Aov) -> Option<&wgpu::Texture> {
        self.companions.get(at)?.texture(aov)
    }

    /// Display image `at`'s film-size copies, the beauty first.
    pub fn film_images(&self, at: usize) -> Vec<(Aov, &wgpu::Texture)> {
        self.companions
            .get(at)
            .map(|c| c.all().collect())
            .unwrap_or_default()
    }

    /// An output's film-resolution texture (Rgba16Float for colors, the
    /// volumes and the normal, R32Float for AO and depth, R32Uint for ids).
    pub fn aov_texture(&self, aov: Aov) -> Option<&wgpu::Texture> {
        Some(&self.targets.as_ref()?.of(aov).texture)
    }

    /// An output of the last frame, read back (waits for the GPU).
    pub fn read_aov(&self, aov: Aov) -> Option<AovImage> {
        let texture = self.aov_texture(aov)?;
        Some(readback::read(&self.device, &self.queue, texture))
    }

    /// Display image `at` of the last frame written to it, read back.
    pub fn read_image(&self, at: usize) -> Option<AovImage> {
        let texture = self.image(at)?;
        Some(readback::read(&self.device, &self.queue, texture))
    }

    /// Display image `at`'s half floats (their bits), RGBA, rows from the
    /// top.
    pub fn read_halves(&self, at: usize) -> Option<Vec<u16>> {
        let texture = self.image(at)?;
        Some(readback::halves(&self.device, &self.queue, texture))
    }

    /// The last submitted frame's GPU time per pass (waits for the GPU);
    /// none when the device cannot time passes.
    pub fn read_timings(&self) -> Option<PassTimes> {
        self.timer.as_ref()?.read(&self.device)
    }

    /// Whether the device times passes.
    pub fn timed(&self) -> bool {
        self.timer.is_some()
    }
}

impl Renderer {
    /// A renderer of an empty scene on `device` (the host's own; wgpu
    /// handles are shared, not copied).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Renderer {
        let scene = GpuScene::new(device);
        Renderer {
            device: device.clone(),
            queue: queue.clone(),
            settings: Settings::default(),
            geometry: Geometry::build(&[]),
            media: Media::build(&[]),
            materials: Vec::new(),
            lights: 0,
            dome: None,
            rims: Rims::default(),
            fog: None,
            gbuffer: GbufferPass::new(device),
            lighting: LightingPasses::new(device, &scene.bvh_layout),
            volume: VolumePass::new(device, &scene.bvh_layout),
            shade: ShadePasses::new(device),
            scene,
            generation: 0,
            geometry_builds: 0,
        }
    }

    /// Replaces the whole scene.
    pub fn set_scene(&mut self, scene: &Scene) {
        self.materials = scene.materials.clone();
        self.set_meshes(&scene.meshes);
        self.set_volumes(&scene.volumes);
        self.set_lights(&scene.lighting());
        self.set_fog(scene.fog);
    }

    /// New meshes: the vertices and the BVH are rebuilt.
    pub fn set_meshes(&mut self, meshes: &[Mesh]) {
        self.geometry = Geometry::build(meshes);
        self.geometry_builds += 1;
        self.scene.set_geometry(&self.device, &self.geometry);
        self.generation += 1;
        self.write_objects();
    }

    /// New volumes (their bounds or media changed): their small BVH is
    /// rebuilt; the surfaces' is not.
    pub fn set_volumes(&mut self, volumes: &[Volume]) {
        self.media = Media::build(volumes);
        self.scene.set_media(&self.device, &self.media);
        self.generation += 1;
    }

    /// New materials (meshes index them).
    pub fn set_materials(&mut self, materials: &[PixelSurface]) {
        self.materials = materials.to_vec();
        self.write_objects();
    }

    /// New lights, domes and rims.
    pub fn set_lights(&mut self, lighting: &Lighting) {
        let packed: Vec<GpuLight> = lighting.lights.iter().map(pack_light).collect();
        if self.scene.set_lights(&self.device, &self.queue, &packed) {
            self.generation += 1;
        }
        self.lights = packed.len() as u32;
        self.dome = lighting.dome;
        self.rims = lighting.rims;
    }

    /// New fog (none: clear air).
    pub fn set_fog(&mut self, fog: Option<Fog>) {
        self.fog = fog;
    }

    /// New settings.
    pub fn set_settings(&mut self, settings: Settings) {
        self.settings = settings.clamped();
    }

    /// The settings in use.
    pub fn settings(&self) -> Settings {
        self.settings
    }

    /// How many times geometry has been built (each [`Renderer::set_meshes`]).
    pub fn geometry_builds(&self) -> u64 {
        self.geometry_builds
    }

    /// The film a `width` × `height` view renders at with the settings in
    /// use.
    pub fn film(&self, width: u32, height: u32) -> Film {
        Film::new(width, height, self.settings.pixel_size)
    }

    /// The device it renders on.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue it submits to.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    fn write_objects(&mut self) {
        let table = self.geometry.object_table(&self.materials);
        if self.scene.set_objects(&self.device, &self.queue, &table) {
            self.generation += 1;
        }
    }

    /// Records a frame of `view` into `viewport`'s display image `image`
    /// and its film, into `encoder` (the host submits it); the frame
    /// uniform is written to the queue, ahead of that submission. Returns
    /// the film, or none for an empty view.
    pub fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &mut Viewport,
        view: &View,
        image: usize,
    ) -> Option<Film> {
        let film = self.film(view.width, view.height);
        if film.is_empty() {
            return None;
        }
        if viewport.targets.as_ref().is_none_or(|t| t.film != film) {
            let targets = Targets::new(&self.device, film);
            viewport.images = (0..viewport.count)
                .map(|_| Targets::image(&self.device, film))
                .collect();
            viewport.companions = (0..viewport.count)
                .map(|_| Companions::new(&self.device, &targets))
                .collect();
            viewport.targets = Some(targets);
            viewport.images_id = IMAGE_SETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            viewport.groups = None;
        }
        let camera = CameraFrame::new(&view.camera, &film, self.geometry.bounds());
        let uniform = FrameUniform::new(&FrameInputs {
            camera: &camera,
            view: film.view,
            settings: &self.settings,
            dome: self.dome,
            rims: self.rims,
            fog: self.fog,
            lights: self.lights,
            extent: self.geometry.extent(),
            volumes: (self.media.count, self.media.casters),
        });
        self.queue
            .write_buffer(&viewport.frame, 0, bytemuck::bytes_of(&uniform));
        let targets = viewport.targets.as_ref()?;
        if viewport
            .groups
            .as_ref()
            .is_none_or(|g| g.generation != self.generation)
        {
            let frame = &viewport.frame;
            viewport.groups = Some(Groups {
                generation: self.generation,
                gbuffer: self.gbuffer.bind(&self.device, frame, &self.scene),
                lighting: self
                    .lighting
                    .bind(&self.device, frame, targets, &self.scene),
                volume: self.volume.bind(&self.device, frame, targets, &self.scene),
                shade: self.shade.bind(&self.device, frame, targets),
                upscale: viewport
                    .images
                    .iter()
                    .map(|i| self.shade.bind_upscale(&self.device, frame, targets, i))
                    .collect(),
            });
        }
        let groups = viewport.groups.as_ref()?;
        let upscale = groups.upscale.get(image)?;
        let timer = viewport.timer.as_ref();
        self.gbuffer
            .encode(encoder, targets, &self.scene, &groups.gbuffer, timer);
        self.lighting
            .encode(encoder, targets, &self.scene, &groups.lighting, timer);
        self.volume
            .encode(encoder, targets, &self.scene, &groups.volume, timer);
        self.shade
            .encode(encoder, targets, &groups.shade, upscale, timer);
        if let Some(t) = timer {
            t.resolve(encoder);
        }
        if let Some(c) = viewport.companions.get_mut(image) {
            c.encode(&self.device, encoder, targets, &viewport.kept);
        }
        Some(film)
    }

    /// Records and submits a frame of `view` into `viewport`'s display
    /// image `image`.
    pub fn render(&mut self, viewport: &mut Viewport, view: &View, image: usize) -> Option<Film> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pixel frame"),
            });
        let film = self.encode(&mut encoder, viewport, view, image)?;
        self.queue.submit([encoder.finish()]);
        Some(film)
    }
}
