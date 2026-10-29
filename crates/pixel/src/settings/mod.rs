//! Pixel Render Settings: the descriptors hosts show ([`descriptors`]) and
//! the typed values a frame renders with ([`Settings`]), read from a host's
//! `kinogaki_render_api::RenderSettings` by name.

mod descriptors;
pub mod ramp;

use kinogaki_render_api::{Choice, RenderSettings, SettingValue};

pub use descriptors::{descriptors, groups};
pub use ramp::Ramp;

use descriptors::key;

/// The ordered dither between adjacent shade levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Dither {
    /// A 2×2 Bayer matrix.
    Bayer2,
    /// A 4×4 Bayer matrix.
    #[default]
    Bayer4,
    /// An 8×8 Bayer matrix.
    Bayer8,
    /// No dither: each pixel takes the nearest level.
    Off,
}

impl Dither {
    /// Every dither, in the order a UI lists them.
    pub const ALL: [Dither; 4] = [Dither::Bayer2, Dither::Bayer4, Dither::Bayer8, Dither::Off];

    /// The matrix's side, 0 when off.
    pub fn size(self) -> u32 {
        match self {
            Dither::Bayer2 => 2,
            Dither::Bayer4 => 4,
            Dither::Bayer8 => 8,
            Dither::Off => 0,
        }
    }

    /// Its choice: its name in files and its label.
    pub fn choice(self) -> Choice {
        match self {
            Dither::Bayer2 => Choice::new("bayer2", "Bayer 2×2"),
            Dither::Bayer4 => Choice::new("bayer4", "Bayer 4×4"),
            Dither::Bayer8 => Choice::new("bayer8", "Bayer 8×8"),
            Dither::Off => Choice::new("off", "Off"),
        }
    }
}

/// The most bounce rays a film pixel traces.
pub const MAX_GI_SAMPLES: u32 = 64;
/// The most steps a ray takes through volumes (and as many through lit
/// fog).
pub const MAX_VOLUME_STEPS: u32 = 256;
/// AO Power's range.
const MIN_AO_POWER: f32 = 0.1;
const MAX_AO_POWER: f32 = 8.0;

/// How a frame's light is quantized and dithered, and how far its
/// occlusion, bounce and volume rays look.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// View pixels a film pixel covers, along each axis (at least 1).
    pub pixel_size: u32,
    /// Shade levels between the darkest and full light (at least 2).
    pub shade_levels: u32,
    /// The dither between levels.
    pub dither: Dither,
    /// Scales the dither's thresholds around one half: 0 rounds to the
    /// nearest level, 1 spreads the whole step.
    pub dither_strength: f32,
    /// The color each level takes.
    pub shade_ramp: Ramp,
    /// How far AO rays look, in scene units.
    pub ao_distance: f32,
    /// How dark occlusion goes: AO is `mix(1, ao^power, strength)`.
    pub ao_strength: f32,
    /// AO's contrast curve: the open share is raised to this power before
    /// strength (1: as traced; above 1 darkens partly open places).
    pub ao_power: f32,
    /// How much AO also darkens direct light, `mix(1, ao, ao_direct)` (a
    /// stylization: contact darkening in dense canopies).
    pub ao_direct: f32,
    /// Scales one bounce of diffuse light off nearby surfaces (0: none).
    pub gi: f32,
    /// Rays a film pixel traces for the bounce: a fixed pattern.
    pub gi_samples: u32,
    /// How far bounce rays look, in scene units.
    pub gi_distance: f32,
    /// Steps a ray takes through the volumes it crosses (and as many
    /// through lit fog), each at a fixed jitter a film pixel.
    pub volume_steps: u32,
    /// How far from the camera volumes and lit fog are marched, scene
    /// units.
    pub volume_max_distance: f32,
    /// Whether each volume step traces a shadow ray to each light (the
    /// shafts) and one toward the sky (ambient occlusion); off, every step
    /// sees every light and the whole sky.
    pub volume_shadow_rays: bool,
    /// Levels of the volumes' and fog's opacity, and of the volumes' own
    /// light through the Shade Ramp (at least 2): their own bands, finer
    /// than the surfaces' shade levels.
    pub volume_levels: u32,
    /// The dither's strength on the volumes and fog: below 1 only the
    /// bands' edges dither, so mist reads as soft bands, not noise.
    pub volume_dither: f32,
    /// The world's up direction (the host's, unit).
    pub up: [f32; 3],
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            pixel_size: 4,
            shade_levels: 4,
            dither: Dither::Bayer4,
            dither_strength: 1.0,
            shade_ramp: Ramp::new(&ramp::default_stops()),
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
            volume_levels: 12,
            volume_dither: 0.25,
            up: [0.0, 1.0, 0.0],
        }
    }
}

fn chosen<T: Copy + Default>(value: SettingValue, all: &[T], choice: impl Fn(T) -> Choice) -> T {
    match value {
        SettingValue::Choice(name) => all
            .iter()
            .copied()
            .find(|v| choice(*v).name == name)
            .unwrap_or_default(),
        _ => T::default(),
    }
}

impl Settings {
    /// The settings `render`'s values give, by name (each missing or
    /// unfitting one at its descriptor's default), and its up direction.
    pub fn from_render(render: &RenderSettings) -> Settings {
        let descriptors = descriptors();
        let value = |name: &str| descriptors::read(&descriptors, render, name);
        let int = |name| match value(name) {
            SettingValue::Int(v) => v as u32,
            _ => 0,
        };
        let float = |name| match value(name) {
            SettingValue::Float(v) => v as f32,
            _ => 0.0,
        };
        let flag = |name| matches!(value(name), SettingValue::Bool(true));
        let shade_ramp = match value(key::SHADE_RAMP) {
            SettingValue::ColorRamp(stops) => Ramp::new(&stops),
            _ => Ramp::grey(),
        };
        Settings {
            pixel_size: int(key::PIXEL_SIZE),
            shade_levels: int(key::SHADE_LEVELS),
            dither: chosen(value(key::DITHER), &Dither::ALL, Dither::choice),
            dither_strength: float(key::DITHER_STRENGTH),
            shade_ramp,
            ao_distance: float(key::AO_DISTANCE),
            ao_strength: float(key::AO_STRENGTH),
            ao_power: float(key::AO_POWER),
            ao_direct: float(key::AO_DIRECT),
            gi: float(key::GI),
            gi_samples: int(key::GI_SAMPLES),
            gi_distance: float(key::GI_DISTANCE),
            volume_steps: int(key::VOLUME_STEPS),
            volume_max_distance: float(key::VOLUME_MAX_DISTANCE),
            volume_shadow_rays: flag(key::VOLUME_SHADOW_RAYS),
            volume_levels: int(key::VOLUME_LEVELS),
            volume_dither: float(key::VOLUME_DITHER),
            up: render.up,
        }
        .clamped()
    }

    /// The settings with every field in its valid range.
    pub fn clamped(&self) -> Settings {
        let up = crate::math::Vec3::new(
            f64::from(self.up[0]),
            f64::from(self.up[1]),
            f64::from(self.up[2]),
        )
        .normalize_or(crate::math::Vec3::Y);
        Settings {
            pixel_size: self.pixel_size.max(1),
            shade_levels: self.shade_levels.max(2),
            dither_strength: self.dither_strength.clamp(0.0, 1.0),
            ao_distance: self.ao_distance.max(0.0),
            ao_strength: self.ao_strength.clamp(0.0, 1.0),
            ao_power: self.ao_power.clamp(MIN_AO_POWER, MAX_AO_POWER),
            ao_direct: self.ao_direct.clamp(0.0, 1.0),
            gi: self.gi.max(0.0),
            gi_samples: self.gi_samples.clamp(1, MAX_GI_SAMPLES),
            gi_distance: self.gi_distance.max(0.0),
            volume_steps: self.volume_steps.clamp(1, MAX_VOLUME_STEPS),
            volume_max_distance: self.volume_max_distance.max(0.0),
            volume_levels: self.volume_levels.max(2),
            volume_dither: self.volume_dither.clamp(0.0, 1.0),
            up: [up.x as f32, up.y as f32, up.z as f32],
            ..*self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_descriptors_default_to_the_settings_defaults() {
        let defaults = RenderSettings {
            values: kinogaki_render_api::settings::defaults(&descriptors()),
            ..RenderSettings::default()
        };
        assert_eq!(Settings::from_render(&defaults), Settings::default());
        assert_eq!(
            Settings::from_render(&RenderSettings::default()),
            Settings::default()
        );
        // Every descriptor sits in a group.
        let groups = groups();
        assert!(
            descriptors()
                .iter()
                .all(|d| groups.iter().any(|g| g.name == d.group))
        );
    }

    #[test]
    fn values_are_read_by_name_and_unfitting_ones_fall_back() {
        let mut render = RenderSettings::default();
        render.set("pixel_size", SettingValue::Int(2));
        render.set("dither", SettingValue::Choice(std::sync::Arc::from("off")));
        render.set("ao_distance", SettingValue::Float(0.25));
        render.set("shade_levels", SettingValue::Int(1000));
        render.set("volume_levels", SettingValue::Int(1));
        render.up = [0.0, 0.0, 2.0];
        let s = Settings::from_render(&render);
        assert_eq!(s.volume_levels, 12, "out of range: the default");
        assert_eq!(s.up, [0.0, 0.0, 1.0]);
        assert_eq!(s.pixel_size, 2);
        assert_eq!(s.dither, Dither::Off);
        assert_eq!(s.ao_distance, 0.25);
        assert_eq!(s.shade_levels, 4, "out of range: the default");
        render.set("gi_samples", SettingValue::Int(8));
        render.set("ao_power", SettingValue::Float(2.0));
        render.set("ao_strength", SettingValue::Float(2.0));
        let s = Settings::from_render(&render);
        assert_eq!((s.gi_samples, s.ao_power), (8, 2.0));
        assert_eq!(s.ao_strength, 1.0, "out of range: the default");
        let wild = Settings {
            gi_samples: 1000,
            ao_power: 0.0,
            ao_direct: -1.0,
            gi: -2.0,
            ..Settings::default()
        }
        .clamped();
        assert_eq!(wild.gi_samples, MAX_GI_SAMPLES);
        render.set("volume_steps", SettingValue::Int(8));
        let s = Settings::from_render(&render);
        assert!(s.volume_shadow_rays);
        assert_eq!(s.volume_steps, 8);
        assert_eq!((wild.ao_power, wild.ao_direct, wild.gi), (0.1, 0.0, 0.0));
    }
}
