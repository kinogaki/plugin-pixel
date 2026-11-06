//! A view's textures: the G-buffer and the light passes at film resolution,
//! and its view-size display images, each with film-size copies of the
//! outputs a compositing host keeps. Remade only when the film's size
//! changes.
use crate::outputs::Output;
use crate::render::film::Film;
use kinogaki_render_api::display::wgpu;

/// Colors, normals and the volumes: albedo, normal, transmittance, direct,
/// ambient, indirect, volume, beauty and the display images.
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
        [width, height]: [u32; 2],
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Target {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
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

/// The format of output `output`'s texture.
pub fn format_of(output: Output) -> wgpu::TextureFormat {
    match output {
        Output::Occlusion | Output::Depth => SCALAR,
        Output::ObjectId => ID,
        _ => COLOR,
    }
}

/// Every texture a frame writes at film resolution.
pub struct Targets {
    /// The film they are sized for.
    pub film: Film,
    /// Each output's texture, in [`Output::ALL`]'s order.
    outputs: Vec<Target>,
    /// The G-buffer's transmittance: what light on a back face is
    /// multiplied by.
    pub transmit: Target,
    /// The rasterizer's depth test.
    pub depth_test: Target,
}

impl Targets {
    /// The textures for `film`.
    pub fn new(device: &wgpu::Device, film: Film) -> Targets {
        use wgpu::TextureUsages as U;
        let raster = U::RENDER_ATTACHMENT | U::TEXTURE_BINDING | U::COPY_SRC;
        let compute = U::STORAGE_BINDING | U::TEXTURE_BINDING | U::COPY_SRC;
        let size = film.size;
        let outputs = (Output::ALL.iter())
            .map(|&o| {
                let usage = if o.rasterized() { raster } else { compute };
                Target::new(device, o.label(), size, format_of(o), usage)
            })
            .collect();
        Targets {
            film,
            outputs,
            transmit: Target::new(device, "pixel transmit", size, COLOR, raster),
            depth_test: Target::new(
                device,
                "pixel depth test",
                size,
                DEPTH_TEST,
                U::RENDER_ATTACHMENT,
            ),
        }
    }

    /// Output `output`'s texture.
    pub fn of(&self, output: Output) -> &Target {
        &self.outputs[output as usize]
    }

    /// Output `output`'s view.
    pub fn view(&self, output: Output) -> &wgpu::TextureView {
        &self.of(output).view
    }
}

/// One display image: the beauty at view size, and film-size copies of the
/// frame's outputs a compositing host keeps, the beauty first.
pub struct Image {
    /// The upscaled beauty.
    pub shown: Target,
    /// The film-size copies, by output.
    pub film: Vec<(Output, Target)>,
}

impl Image {
    /// A display image for `film`'s view keeping `kept` beside the beauty.
    pub fn new(device: &wgpu::Device, film: Film, kept: &[Output]) -> Image {
        use wgpu::TextureUsages as U;
        let usage = U::STORAGE_BINDING | U::TEXTURE_BINDING | U::COPY_SRC;
        let copy = U::COPY_DST | U::TEXTURE_BINDING | U::COPY_SRC;
        let outputs = [Output::Beauty].into_iter().chain(kept.iter().copied());
        Image {
            shown: Target::new(device, "pixel shown", film.view, COLOR, usage),
            film: outputs
                .map(|o| {
                    (
                        o,
                        Target::new(device, o.label(), film.size, format_of(o), copy),
                    )
                })
                .collect(),
        }
    }

    /// Records the copies of `targets`' outputs into its film images.
    pub fn copy(&self, encoder: &mut wgpu::CommandEncoder, targets: &Targets) {
        let [width, height] = targets.film.size;
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        for (output, image) in &self.film {
            encoder.copy_texture_to_texture(
                targets.of(*output).texture.as_image_copy(),
                image.texture.as_image_copy(),
                size,
            );
        }
    }
}
