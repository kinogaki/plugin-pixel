//! Pixel Render Settings as descriptors hosts show, in four groups:
//! Pixels (the grid, levels, dither and Shade Ramp), Ambient Occlusion
//! (how far AO looks and how it darkens), Global Illumination (one
//! diffuse bounce) and Volumes (how volumes and lit fog are marched, and
//! their own opacity levels and dither). Light, ambient light, rim light
//! and fog come from the scene's lights and Fog prim.

use std::sync::Arc;

use kinogaki_render_api::{
    Choice, ColorStop, SettingDescriptor, SettingGroup, SettingKind, SettingValue, Unit,
};

use super::{Dither, Settings};

/// The descriptors' names.
pub(super) mod key {
    pub const PIXEL_SIZE: &str = "pixel_size";
    pub const SHADE_LEVELS: &str = "shade_levels";
    pub const DITHER: &str = "dither";
    pub const DITHER_STRENGTH: &str = "dither_strength";
    pub const SHADE_RAMP: &str = "shade_ramp";
    pub const AO_DISTANCE: &str = "ao_distance";
    pub const AO_STRENGTH: &str = "ao_strength";
    pub const AO_POWER: &str = "ao_power";
    pub const AO_DIRECT: &str = "ao_direct";
    pub const GI: &str = "gi";
    pub const GI_SAMPLES: &str = "gi_samples";
    pub const GI_DISTANCE: &str = "gi_distance";
    pub const VOLUME_STEPS: &str = "volume_steps";
    pub const VOLUME_MAX_DISTANCE: &str = "volume_max_distance";
    pub const VOLUME_SHADOW_RAYS: &str = "volume_shadow_rays";
    pub const VOLUME_LEVELS: &str = "volume_levels";
    pub const VOLUME_DITHER: &str = "volume_dither";
}

/// The groups' names.
const PIXELS: &str = "pixels";
const OCCLUSION: &str = "occlusion";
const BOUNCE: &str = "gi";
const VOLUMES: &str = "volumes";

/// The settings' groups, in order.
pub fn groups() -> Vec<SettingGroup> {
    let group = |name, label| SettingGroup::new(name, label, true);
    vec![
        group(PIXELS, "Pixels"),
        group(OCCLUSION, "Ambient Occlusion"),
        group(BOUNCE, "Global Illumination"),
        group(VOLUMES, "Volumes"),
    ]
}

fn row(name: &str, label: &str, group: &str, kind: SettingKind) -> SettingDescriptor {
    SettingDescriptor::new(name, label, group, kind)
}

fn int(default: u32, min: i64, max: i64) -> SettingKind {
    SettingKind::Int {
        default: i64::from(default),
        min,
        max,
    }
}

/// A number in `min..=max`: a slider over `soft`, or, a `length` in scene
/// units, a field dragged 0.01 a pixel.
fn float(default: f32, (min, max): (f64, f64), soft: (f64, f64), length: bool) -> SettingKind {
    SettingKind::Float {
        default: f64::from(default),
        min,
        max,
        soft,
        decimals: if length { 2 } else { 3 },
        unit: if length { Unit::Length } else { Unit::None },
        slider: !length,
        step: 0.01,
    }
}

fn choice(choices: Vec<Choice>, default: Choice) -> SettingKind {
    SettingKind::Choice {
        choices,
        default: default.name,
        menu: false,
    }
}

fn tip(d: SettingDescriptor, tooltip: &str) -> SettingDescriptor {
    d.tooltip(tooltip)
}

/// Pixel Render's settings, with the owner's defaults.
pub fn descriptors() -> Vec<SettingDescriptor> {
    use key::*;
    let d = Settings::default();
    let ramp: Arc<[ColorStop]> = Arc::from(super::ramp::default_stops());
    let strength = (0.0, 1000.0);
    vec![
        // Pixels.
        row(PIXEL_SIZE, "Pixel Size", PIXELS, int(d.pixel_size, 1, 64)),
        row(
            SHADE_LEVELS,
            "Shade Levels",
            PIXELS,
            int(d.shade_levels, 2, 64),
        ),
        row(
            DITHER,
            "Dither",
            PIXELS,
            choice(Dither::ALL.map(Dither::choice).to_vec(), d.dither.choice()),
        ),
        row(
            DITHER_STRENGTH,
            "Dither Strength",
            PIXELS,
            float(d.dither_strength, (0.0, 1.0), (0.0, 1.0), false),
        ),
        tip(
            row(
                SHADE_RAMP,
                "Shade Ramp",
                PIXELS,
                SettingKind::ColorRamp { default: ramp },
            ),
            "The color of each shade level, darkest to brightest",
        ),
        // Ambient Occlusion.
        tip(
            row(
                AO_DISTANCE,
                "AO Distance",
                OCCLUSION,
                float(d.ao_distance, (0.0, 1e6), (0.0, 10.0), true),
            ),
            "How far ambient occlusion looks",
        ),
        tip(
            row(
                AO_STRENGTH,
                "AO Strength",
                OCCLUSION,
                float(d.ao_strength, (0.0, 1.0), (0.0, 1.0), false),
            ),
            "How dark occluded places go: 0 ignores AO, 1 applies all of it",
        ),
        tip(
            row(
                AO_POWER,
                "AO Power",
                OCCLUSION,
                float(
                    d.ao_power,
                    (
                        f64::from(super::MIN_AO_POWER),
                        f64::from(super::MAX_AO_POWER),
                    ),
                    (0.25, 4.0),
                    false,
                ),
            ),
            "AO's contrast: above 1 darkens partly open places more",
        ),
        tip(
            row(
                AO_DIRECT,
                "AO on Direct",
                OCCLUSION,
                float(d.ao_direct, (0.0, 1.0), (0.0, 1.0), false),
            ),
            "How much AO also darkens direct light, for contact shadows in dense foliage",
        ),
        // Global Illumination.
        tip(
            row(GI, "GI", BOUNCE, float(d.gi, strength, (0.0, 4.0), false)),
            "Scales one bounce of light off nearby surfaces; 0 turns it off",
        ),
        tip(
            row(
                GI_SAMPLES,
                "GI Samples",
                BOUNCE,
                int(d.gi_samples, 1, i64::from(super::MAX_GI_SAMPLES)),
            ),
            "Bounce rays a pixel traces, in a fixed pattern",
        ),
        tip(
            row(
                GI_DISTANCE,
                "GI Distance",
                BOUNCE,
                float(d.gi_distance, (0.0, 1e6), (0.0, 20.0), true),
            ),
            "How far bounce rays look",
        ),
        // Volumes.
        tip(
            row(
                VOLUME_STEPS,
                "Volume Steps",
                VOLUMES,
                int(d.volume_steps, 1, i64::from(super::MAX_VOLUME_STEPS)),
            ),
            "Steps a ray takes through volumes, and as many through lit fog",
        ),
        tip(
            row(
                VOLUME_MAX_DISTANCE,
                "Volume Max Distance",
                VOLUMES,
                float(d.volume_max_distance, (0.0, 1e6), (0.0, 500.0), true),
            ),
            "How far from the camera volumes and lit fog are marched",
        ),
        tip(
            row(
                VOLUME_SHADOW_RAYS,
                "Volume Shadow Rays",
                VOLUMES,
                SettingKind::Bool {
                    default: d.volume_shadow_rays,
                },
            ),
            "Each step traces shadow rays: light shafts and a darker inside under cover",
        ),
        tip(
            row(
                VOLUME_LEVELS,
                "Volume Levels",
                VOLUMES,
                int(d.volume_levels, 2, 64),
            ),
            "Levels of the opacity and light of volumes and fog",
        ),
        tip(
            row(
                VOLUME_DITHER,
                "Volume Dither",
                VOLUMES,
                float(d.volume_dither, (0.0, 1.0), (0.0, 1.0), false),
            ),
            "Dither on volumes and fog: below 1 only the bands' edges dither",
        ),
    ]
}

/// Setting `name`'s value in `values`, or its default when missing or
/// unfitting.
pub(super) fn read(
    descriptors: &[SettingDescriptor],
    render: &kinogaki_render_api::RenderSettings,
    name: &str,
) -> SettingValue {
    let d = descriptors
        .iter()
        .find(|d| &*d.name == name)
        .expect("a Pixel Render setting");
    render
        .value(name)
        .filter(|v| d.accepts(v))
        .cloned()
        .unwrap_or_else(|| d.default_value())
}
