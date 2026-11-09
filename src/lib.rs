//! Pixel, a realtime pixel-art render engine, as a Kinogaki Studio plugin.
//!
//! A frame renders at a coarse film resolution, `ceil(view / pixel size)`,
//! on the host's own wgpu device: a G-buffer, ray-traced ambient occlusion,
//! hard shadows and one bounce of diffuse light, volumes, then the light
//! quantized by its luminance and dithered through the Shade Ramp, and each
//! film pixel repeated over its block of the view. Inputs and outputs are
//! linear ACEScg.
//!
//! The renderer knows no SDK: [`render`] (the scene, meshes in their own
//! space placed by instances and the rest in world space; film, camera, the
//! two levels of BVH, dither) and [`gpu`] (the passes and their shaders). The plugin around it: the package, the engine's
//! [`description`] and [`settings`], its material [`nodes`] and starting
//! materials ([`presets`]), its [`lights`] (the standard ones it renders
//! and fog are core's objects; its Rim Light its own), its [`outputs`], scene
//! [`sync`] with the [`pictures`] textures become, and the render thread
//! ([`engine`]).
pub mod description;
pub mod engine;
pub mod gpu;
pub mod lights;
pub mod nodes;
pub mod outputs;
pub mod pictures;
pub mod presets;
pub mod render;
pub mod sampled;
pub mod settings;
pub mod shadowless;
pub mod sync;

use kinogaki_studio_api::{Package, PackageId};

/// Pixel's package: the engine with its material nodes and its Render
/// Output, its Rim Light, and its starting materials, each in its family.
pub fn package() -> Package {
    let package = Package::new(PackageId::declared("kinogaki.pixel"), "Pixel")
        .with_engine(engine::Pixel)
        .with_object_kind(lights::lights());
    (presets::creations().into_iter()).fold(package, Package::with_creation)
}
