//! Pixel's render settings, as a Render Output shows them ([`descriptors`],
//! by [`groups`]) and as a frame renders them ([`read`]): Pixels (Pixel
//! Size, Shade Levels, the dither and the Shade Ramp), Ambient Occlusion,
//! Global Illumination and Volumes.
use crate::render::dither::Dither;
use crate::render::settings::{MAX_GI_SAMPLES, MAX_LEVELS, MAX_VOLUME_STEPS, Settings};
use kinogaki_render_api::settings::fit;
use kinogaki_render_api::value::ramp::{Interpolation, Key};
use kinogaki_render_api::value::{Choice, Number, NumberLook, Parameters, Ramp, Value, ValueKind};
use kinogaki_render_api::{SettingDescriptor, SettingGroup, standard};

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

/// The dithers, by key, as the Dither row offers them.
const DITHERS: [(Dither, &str, &str); 4] = [
    (Dither::Bayer2, "bayer2", "Bayer 2×2"),
    (Dither::Bayer4, "bayer4", "Bayer 4×4"),
    (Dither::Bayer8, "bayer8", "Bayer 8×8"),
    (Dither::Off, "off", "Off"),
];

/// The Shade Ramp's default: deep blue shade up to warm light.
fn default_ramp() -> Ramp {
    let key = |p, rgb| Key::color(p, rgb, Interpolation::Linear);
    Ramp {
        keys: vec![
            key(0.0, [0.035, 0.03, 0.09]),
            key(0.34, [0.22, 0.23, 0.36]),
            key(0.67, [0.68, 0.66, 0.62]),
            key(1.0, [1.12, 0.98, 0.76]),
        ],
        color: true,
    }
}

/// An integer row over `min..=max`.
fn integer(
    name: &str,
    label: &str,
    group: &str,
    (min, max): (i64, i64),
    default: i64,
) -> SettingDescriptor {
    SettingDescriptor::new(name, label, group, ValueKind::integer(min, max), default)
}

/// A slider row over `min..=max`, dragged over `soft`.
fn slider(
    name: &str,
    label: &str,
    group: &str,
    (min, max, soft): (f64, f64, (f64, f64)),
    default: f64,
) -> SettingDescriptor {
    let kind = ValueKind::Number(Number::plain().range(min, max).soft(soft.0, soft.1));
    SettingDescriptor::new(name, label, group, kind, default)
        .look(NumberLook::slider().decimals(3).speed(0.01))
}

/// A length row in scene units, dragged over `soft`.
fn length(
    name: &str,
    label: &str,
    group: &str,
    soft: (f64, f64),
    default: f64,
) -> SettingDescriptor {
    let kind = ValueKind::Number(Number::length().range(0.0, 1e6).soft(soft.0, soft.1));
    SettingDescriptor::new(name, label, group, kind, default)
        .look(NumberLook::default().decimals(2).speed(0.01))
}

/// The settings, in the order a Render Output lists them.
pub fn descriptors() -> Vec<SettingDescriptor> {
    let levels = (2, i64::from(MAX_LEVELS));
    let dithers = DITHERS
        .map(|(_, key, label)| Choice::new(key, label))
        .to_vec();
    vec![
        integer("pixel_size", "Pixel Size", PIXELS, (1, 64), 4)
            .tooltip("View pixels a film pixel covers"),
        integer("shade_levels", "Shade Levels", PIXELS, levels, 4),
        SettingDescriptor::new(
            "dither",
            "Dither",
            PIXELS,
            ValueKind::choice(dithers),
            standard::choice("bayer4"),
        ),
        slider(
            "dither_strength",
            "Dither Strength",
            PIXELS,
            (0.0, 1.0, (0.0, 1.0)),
            1.0,
        ),
        SettingDescriptor::new(
            "shade_ramp",
            "Shade Ramp",
            PIXELS,
            ValueKind::Ramp {
                color: true,
                most: None,
            },
            Value::Ramp(default_ramp()),
        )
        .tooltip("The color of each shade level, darkest to brightest"),
        length("ao_distance", "AO Distance", OCCLUSION, (0.0, 10.0), 1.0)
            .tooltip("How far ambient occlusion looks"),
        slider(
            "ao_strength",
            "AO Strength",
            OCCLUSION,
            (0.0, 1.0, (0.0, 1.0)),
            1.0,
        )
        .tooltip("How dark occluded places go: 0 ignores AO, 1 applies all of it"),
        slider(
            "ao_power",
            "AO Power",
            OCCLUSION,
            (0.1, 8.0, (0.25, 4.0)),
            1.0,
        )
        .tooltip("AO's contrast: above 1 darkens partly open places more"),
        slider(
            "ao_direct",
            "AO on Direct",
            OCCLUSION,
            (0.0, 1.0, (0.0, 1.0)),
            0.0,
        )
        .tooltip("How much AO also darkens direct light, for contact shadows in dense foliage"),
        slider("gi", "GI", BOUNCE, (0.0, 1000.0, (0.0, 4.0)), 1.0)
            .tooltip("Scales one bounce of light off nearby surfaces; 0 turns it off"),
        integer(
            "gi_samples",
            "GI Samples",
            BOUNCE,
            (1, i64::from(MAX_GI_SAMPLES)),
            16,
        )
        .tooltip("Bounce rays a pixel traces, in a fixed pattern")
        .animatable(false),
        length("gi_distance", "GI Distance", BOUNCE, (0.0, 20.0), 4.0)
            .tooltip("How far bounce rays look"),
        integer(
            "volume_steps",
            "Volume Steps",
            VOLUMES,
            (1, i64::from(MAX_VOLUME_STEPS)),
            24,
        )
        .tooltip("Steps a ray takes through volumes, and as many through lit fog")
        .animatable(false),
        length(
            "volume_max_distance",
            "Volume Max Distance",
            VOLUMES,
            (0.0, 500.0),
            100.0,
        )
        .tooltip("How far from the camera volumes and lit fog are marched"),
        SettingDescriptor::new(
            "volume_shadow_rays",
            "Volume Shadow Rays",
            VOLUMES,
            ValueKind::Toggle,
            true,
        )
        .tooltip("Each step traces shadow rays: light shafts and a darker inside under cover")
        .animatable(false),
        integer("volume_levels", "Volume Levels", VOLUMES, levels, 12)
            .tooltip("Levels of the opacity and light of volumes and fog"),
        slider(
            "volume_dither",
            "Volume Dither",
            VOLUMES,
            (0.0, 1.0, (0.0, 1.0)),
            0.25,
        )
        .tooltip("Dither on volumes and fog: below 1 only the bands' edges dither"),
    ]
}

/// `ramp` at each of `levels` levels, darkest first.
fn table(ramp: &Ramp, levels: i64) -> Vec<[f32; 3]> {
    let n = levels.clamp(2, i64::from(MAX_LEVELS)) as usize;
    (0..n)
        .map(|k| ramp.evaluate(k as f64 / (n - 1) as f64).map(|c| c as f32))
        .collect()
}

/// The settings `values` give (each missing or unfitting one at its
/// default), with `up`, the world's up direction (unit).
pub fn read(values: &Parameters, up: [f32; 3]) -> Settings {
    let values = fit(&descriptors(), values);
    let number = |name: &str| values.get(name).and_then(Value::number).unwrap_or(0.0) as f32;
    let integer = |name: &str| values.get(name).and_then(Value::integer).unwrap_or(0);
    let count = |name: &str| integer(name).max(1) as u32;
    let key = values.get("dither").and_then(Value::choice).unwrap_or("");
    let ramp = (values.get("shade_ramp").and_then(Value::ramp))
        .cloned()
        .unwrap_or_else(default_ramp);
    Settings {
        pixel_size: count("pixel_size"),
        dither: DITHERS
            .iter()
            .find(|d| d.1 == key)
            .map_or(Dither::Bayer4, |d| d.0),
        dither_strength: number("dither_strength"),
        shade_ramp: table(&ramp, integer("shade_levels")),
        volume_ramp: table(&ramp, integer("volume_levels")),
        volume_dither: number("volume_dither"),
        ao_distance: number("ao_distance"),
        ao_strength: number("ao_strength"),
        ao_power: number("ao_power"),
        ao_direct: number("ao_direct"),
        gi: number("gi"),
        gi_samples: count("gi_samples"),
        gi_distance: number("gi_distance"),
        volume_steps: count("volume_steps"),
        volume_max_distance: number("volume_max_distance"),
        volume_shadow_rays: values
            .get("volume_shadow_rays")
            .and_then(Value::bool)
            .unwrap_or(true),
        up,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UP: [f32; 3] = [0.0, 1.0, 0.0];

    #[test]
    fn the_defaults_read_as_the_shade_ramp_at_four_levels() {
        let s = read(&Parameters::new(), UP);
        assert_eq!(
            (s.pixel_size, s.dither, s.gi_samples),
            (4, Dither::Bayer4, 16)
        );
        assert_eq!(s.shade_ramp.len(), 4);
        assert_eq!(s.shade_ramp[0], [0.035, 0.03, 0.09]);
        assert_eq!(s.shade_ramp[3], [1.12, 0.98, 0.76]);
        assert_eq!(s.volume_ramp.len(), 12);
        let groups = groups();
        assert!(
            descriptors()
                .iter()
                .all(|d| groups.iter().any(|g| g.name == d.group))
        );
    }

    #[test]
    fn values_are_read_by_name_and_unfitting_ones_take_their_default() {
        let values: Parameters = [
            ("pixel_size", Value::Integer(2)),
            ("dither", standard::choice("off")),
            ("shade_levels", Value::Integer(1000)),
            ("ao_distance", Value::Number(0.25)),
            ("gi_samples", Value::Integer(8)),
        ]
        .into_iter()
        .map(|(n, v)| (n.to_owned(), v))
        .collect();
        let s = read(&values, UP);
        assert_eq!(
            (s.pixel_size, s.dither, s.ao_distance, s.gi_samples),
            (2, Dither::Off, 0.25, 8)
        );
        assert_eq!(s.shade_ramp.len(), 4, "out of range: the default");
    }

    #[test]
    fn sample_counts_are_never_keyed() {
        let fixed: Vec<String> = (descriptors().into_iter())
            .filter(|d| !d.animatable)
            .map(|d| d.name.to_string())
            .collect();
        assert_eq!(fixed, ["gi_samples", "volume_steps", "volume_shadow_rays"]);
    }
}
