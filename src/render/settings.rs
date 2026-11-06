//! How a frame's light is quantized and dithered, and how far its
//! occlusion, bounce and volume rays look: a view's settings as the passes
//! read them (the crate's `settings` module reads them from a view's
//! values).
use super::dither::Dither;

/// The most shade levels and volume levels.
pub const MAX_LEVELS: u32 = 64;
/// The most bounce rays a film pixel traces.
pub const MAX_GI_SAMPLES: u32 = 64;
/// The most steps a ray takes through volumes.
pub const MAX_VOLUME_STEPS: u32 = 256;

/// A view's settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// View pixels a film pixel covers, along each axis (at least 1).
    pub pixel_size: u32,
    /// The dither between levels.
    pub dither: Dither,
    /// Scales the dither's thresholds around one half: 0 rounds to the
    /// nearest level, 1 spreads the whole step.
    pub dither_strength: f32,
    /// The Shade Ramp at each shade level, darkest first: as many colors as
    /// levels (2 to [`MAX_LEVELS`]).
    pub shade_ramp: Vec<[f32; 3]>,
    /// The Shade Ramp at each of the volumes' and fog's own levels.
    pub volume_ramp: Vec<[f32; 3]>,
    /// The dither's strength on the volumes and fog: below 1 only the
    /// bands' edges dither.
    pub volume_dither: f32,
    /// How far AO rays look, scene units.
    pub ao_distance: f32,
    /// How dark occlusion goes: AO is `mix(1, open^power, strength)`.
    pub ao_strength: f32,
    /// AO's contrast curve.
    pub ao_power: f32,
    /// How much AO also darkens direct light.
    pub ao_direct: f32,
    /// Scales one bounce of diffuse light (0: none).
    pub gi: f32,
    /// Bounce rays a film pixel traces, in a fixed pattern.
    pub gi_samples: u32,
    /// How far bounce rays look, scene units.
    pub gi_distance: f32,
    /// Steps a ray takes through volumes (and as many through lit fog).
    pub volume_steps: u32,
    /// How far from the camera volumes and lit fog are marched.
    pub volume_max_distance: f32,
    /// Each volume step traces shadow rays and a sky ray.
    pub volume_shadow_rays: bool,
    /// The world's up direction, unit (the host's up axis).
    pub up: [f32; 3],
}

impl Default for Settings {
    /// Four grey levels at four pixels a film pixel.
    fn default() -> Settings {
        let grey = |n: u32| (0..n).map(|k| [k as f32 / (n - 1) as f32; 3]).collect();
        Settings {
            pixel_size: 4,
            dither: Dither::Bayer4,
            dither_strength: 1.0,
            shade_ramp: grey(4),
            volume_ramp: grey(12),
            volume_dither: 0.25,
            ao_distance: 1.0,
            ao_strength: 1.0,
            ao_power: 1.0,
            ao_direct: 0.0,
            gi: 1.0,
            gi_samples: 16,
            gi_distance: 4.0,
            volume_steps: 24,
            volume_max_distance: 100.0,
            volume_shadow_rays: true,
            up: [0.0, 1.0, 0.0],
        }
    }
}
