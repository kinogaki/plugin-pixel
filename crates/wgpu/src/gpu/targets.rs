//! A view's textures: the G-buffer and the lighting and beauty AOVs at film
//! resolution, and its view-size display images. Remade only when the film
//! or view size changes.

use kinogaki_pixel_render::{Aov, Film};

/// Albedo, normal, transmittance, direct, ambient, indirect, volume, beauty
/// and the view images.
pub const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Depth and AO.
pub const SCALAR: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Object ids.
pub const ID: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
/// The G-buffer's depth test.
pub const DEPTH_TEST: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// A texture and its whole view.
pub struct Target {
    /// The texture.
    pub texture: wgpu::Texture,
    /// Its view.
    pub view: wgpu::TextureView,
}

impl Target {
    /// A `size` texture of `format`.
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        size: [u32; 2],
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Target {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size[0].max(1),
                height: size[1].max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Target { texture, view }
    }
}

/// Every texture a frame writes.
pub struct Targets {
    /// The film they are sized for.
    pub film: Film,
    /// G-buffer albedo (ACEScg).
    pub albedo: Target,
    /// G-buffer world normal.
    pub normal: Target,
    /// G-buffer depth along the view direction.
    pub depth: Target,
    /// G-buffer object id.
    pub id: Target,
    /// G-buffer transmittance: what light on the back face is multiplied by.
    pub transmit: Target,
    /// The rasterizer's depth test.
    pub depth_test: Target,
    /// Direct light.
    pub direct: Target,
    /// Ambient light.
    pub ambient: Target,
    /// Indirect light: one diffuse bounce.
    pub indirect: Target,
    /// Ambient occlusion.
    pub ao: Target,
    /// The volumes' light (rgb) and opacity (a), premultiplied.
    pub volume: Target,
    /// The shaded, dithered film.
    pub beauty: Target,
}

impl Targets {
    /// The textures for `film`.
    pub fn new(device: &wgpu::Device, film: Film) -> Targets {
        use wgpu::TextureUsages as U;
        let raster = U::RENDER_ATTACHMENT | U::TEXTURE_BINDING | U::COPY_SRC;
        let compute = U::STORAGE_BINDING | U::TEXTURE_BINDING | U::COPY_SRC;
        let s = film.size;
        Targets {
            film,
            albedo: Target::new(device, "pixel albedo", s, COLOR, raster),
            normal: Target::new(device, "pixel normal", s, COLOR, raster),
            depth: Target::new(device, "pixel depth", s, SCALAR, raster),
            id: Target::new(device, "pixel id", s, ID, raster),
            transmit: Target::new(device, "pixel transmit", s, COLOR, raster),
            depth_test: Target::new(
                device,
                "pixel depth test",
                s,
                DEPTH_TEST,
                U::RENDER_ATTACHMENT,
            ),
            direct: Target::new(device, "pixel direct", s, COLOR, compute),
            ambient: Target::new(device, "pixel ambient", s, COLOR, compute),
            indirect: Target::new(device, "pixel indirect", s, COLOR, compute),
            ao: Target::new(device, "pixel ao", s, SCALAR, compute),
            volume: Target::new(device, "pixel volume", s, COLOR, compute),
            beauty: Target::new(device, "pixel beauty", s, COLOR, compute),
        }
    }

    /// Output `aov`'s texture.
    pub fn of(&self, aov: Aov) -> &Target {
        match aov {
            Aov::Beauty => &self.beauty,
            Aov::Albedo => &self.albedo,
            Aov::Direct => &self.direct,
            Aov::Ambient => &self.ambient,
            Aov::Indirect => &self.indirect,
            Aov::Volume => &self.volume,
            Aov::Ao => &self.ao,
            Aov::Depth => &self.depth,
            Aov::Normal => &self.normal,
            Aov::Id => &self.id,
        }
    }

    /// A display image for `film`'s view: the beauty upscaled (written by
    /// the upscale pass, sampled by a host).
    pub fn image(device: &wgpu::Device, film: Film) -> Target {
        use wgpu::TextureUsages as U;
        let usage = U::STORAGE_BINDING | U::TEXTURE_BINDING | U::COPY_SRC;
        Target::new(device, "pixel view", film.view, COLOR, usage)
    }
}
