//! Pixel Render's host core, Kinogaki Studio's realtime pixel-art
//! renderer without a device: what hosts read before making one
//! ([`description()`]: its settings, the prim types it declares (the
//! standard lights and fog, its [`rim`] light and [`gobo`]), the Pixel
//! Surface and Pixel Volume vocabulary, its outputs), the scene a host's
//! prims make ([`store`]) and the parts of a frame the GPU crate
//! (`kinogaki-pixel-render-wgpu`) shares with its CPU references: the
//! film, the camera, the BVH and the dither.
//!
//! A frame renders at film resolution, `ceil(view / pixel_size)`
//! ([`Film`]), in seven GPU passes:
//!
//! 1. **G-buffer** (raster): albedo (the material's Color times its color
//!    attribute, [`tint`]), normal, depth, object id and the material's
//!    transmittance (Translucency × Translucency Color).
//! 2. **Ambient** (compute): ambient light × ambient occlusion, a fixed
//!    set of hemisphere rays a pixel ([`hemisphere`]) against the [`Bvh`],
//!    up to `ao_distance`, shaped by AO Power and Strength
//!    ([`shading::occlusion`]). The light is the domes' from above and
//!    from below the horizon (a dome's ground color) blended by how far
//!    the surface faces up; no dome, no ambient light.
//! 3. **Direct** (compute): Lambert light from each light ([`lights`]):
//!    area lights and spots as point emitters at their centres, distant
//!    lights as parallel light, each with one hard shadow ray against the
//!    BVH; spots fade over their cone, and a gobo ([`gobo`]) can dapple
//!    distant lights and spots. Light on a translucent surface's back face
//!    comes through × its transmittance, shadowed from the back, and a
//!    shadow ray keeps the product of the transmittances of the thin
//!    surfaces it crosses ([`Bvh::transmittance`]), so light dapples
//!    through foliage. AO on Direct darkens it by the AO.
//! 4. **Indirect** (compute): one diffuse bounce: `gi_samples` rays a
//!    pixel in the same fixed pattern, up to `gi_distance`; each hit's
//!    direct light (as pass 3 computes it) × its albedo (its material's
//!    color × its triangle's mean tint, [`geometry::Surface`]), × `gi`.
//! 5. **Volume** (compute): each camera ray marched through the volumes
//!    it crosses ([`media`]: closed meshes whose material is a Pixel
//!    Volume, found by their crossings in a BVH of their own) and lit fog,
//!    up to the first surface or `volume_max_distance`, `volume_steps`
//!    steps at a fixed jitter a pixel; each step lit by the lights (shadow
//!    rays: surfaces cut light shafts, translucent ones let some
//!    through), the gobos and the ambient light × one sky ray. It writes
//!    their light and opacity, premultiplied.
//! 6. **Shade** (compute): direct + ambient + indirect light and the rim
//!    lights' (the Lit ones only where direct light reaches), quantized by
//!    luminance to `shade_levels` levels with an ordered
//!    Bayer dither between adjacent levels anchored to film pixels
//!    ([`dither`]); each level takes its Shade Ramp color, × albedo; then
//!    the volumes over it ([`shading::over`]: `L_surface · T + L`, the
//!    opacity and the media's light through the ramp on Volume Levels at
//!    Volume Dither, [`shading::Bands`]), then, unless the fog is
//!    lit, height fog, its amount on the same bands ([`shading`]).
//! 7. **Upscale** (compute): the beauty at view size, each film pixel a
//!    `pixel_size`² block from the view's top-left corner, so the grid
//!    stays locked to the pane.
//!
//! Cameras are perspective or orthographic ([`camera`]): an orthographic
//! camera's primary rays are parallel; shadow, AO and bounce rays are the same
//! either way.
//!
//! # Color
//!
//! Everything is linear ACEScg, in and out: material colors, light and dome
//! colors, and every color output (beauty, albedo, direct, ambient,
//! indirect, volume) and the view image. Quantization and dither act on the light
//! before albedo is multiplied in and on the fog's amount, so the beauty is
//! still ACEScg linear; the host's display path (an OCIO view) and Studio's
//! Comp context take it from there. Data outputs (AO, depth, normal, id)
//! are not colors.

pub mod aov;
pub mod bvh;
pub mod camera;
pub mod description;
pub mod dither;
pub mod film;
pub mod geometry;
pub mod gobo;
pub mod hemisphere;
pub mod lights;
pub mod material;
pub mod math;
pub mod media;
pub mod rim;
pub mod scene;
pub mod settings;
pub mod shading;
mod spectrum;
pub mod store;
pub mod tint;
pub mod volume;

pub use aov::{Aov, AovImage, AovPixels};
pub use bvh::Bvh;
pub use description::description;
pub use film::Film;
pub use scene::{
    Camera, Dome, Fog, Gobo, IDENTITY, Light, LightShape, Lighting, Matrix, Mesh, PixelMaterial,
    PixelSurface, Projection, RimSide, Rims, Scene, Volume,
};
pub use settings::{Dither, Ramp, Settings};
pub use store::{Changed, Store};
pub use volume::PixelVolume;
