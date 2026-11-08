//! Pixel's render settings, as a Render Output shows them ([`schema`]) and
//! as a frame renders them ([`read`]): Pixels (Pixel Size, Shade Levels,
//! the dither and the Shade Ramp), Ambient Occlusion, Global Illumination
//! and Volumes.
use crate::render::dither::Dither;
use crate::render::settings::{MAX_GI_SAMPLES, MAX_LEVELS, MAX_VOLUME_STEPS, Settings};
use kinogaki_render_api::value::ramp::{Interpolation, Key};
use kinogaki_render_api::value::{
    Choice, Parameters, ParametersExt, Ramp, Schema, Value, ValueKind,
};
use kinogaki_render_api::{Parameter, Section, standard};

const PIXELS: &str = "pixels";
const OCCLUSION: &str = "occlusion";
const BOUNCE: &str = "gi";
const VOLUMES: &str = "volumes";

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

/// The settings, in the order a Render Output lists them, in their sections
/// (Pixels, Ambient Occlusion, Global Illumination, Volumes; all open).
pub fn schema() -> Schema {
    let most = i64::from(MAX_LEVELS);
    let dithers = DITHERS
        .map(|(_, key, label)| Choice::new(key, label))
        .to_vec();
    let section = |name, label| Section::new(name, label, true);
    let sections = (Schema::default())
        .section(section(PIXELS, "Pixels"))
        .section(section(OCCLUSION, "Ambient Occlusion"))
        .section(section(BOUNCE, "Global Illumination"))
        .section(section(VOLUMES, "Volumes"));
    [
        Parameter::integer("pixel_size", "Pixel Size", 1, 64, 4)
            .group(PIXELS)
            .help("View pixels a film pixel covers"),
        Parameter::integer("shade_levels", "Shade Levels", 2, most, 4).group(PIXELS),
        Parameter::new(
            "dither",
            "Dither",
            ValueKind::choice(dithers),
            standard::choice("bayer4"),
        )
        .group(PIXELS),
        Parameter::slider(
            "dither_strength",
            "Dither Strength",
            (0.0, 1.0),
            (0.0, 1.0),
            3,
            1.0,
        )
        .group(PIXELS),
        Parameter::new(
            "shade_ramp",
            "Shade Ramp",
            ValueKind::Ramp {
                color: true,
                most: None,
            },
            Value::Ramp(default_ramp()),
        )
        .group(PIXELS)
        .help("The color of each shade level, darkest to brightest"),
        Parameter::length(
            "ao_distance",
            "AO Distance",
            (0.0, 1e6),
            Some((0.0, 10.0)),
            2,
            1.0,
        )
        .group(OCCLUSION)
        .help("How far ambient occlusion looks"),
        Parameter::slider("ao_strength", "AO Strength", (0.0, 1.0), (0.0, 1.0), 3, 1.0)
            .group(OCCLUSION)
            .help("How dark occluded places go: 0 ignores AO, 1 applies all of it"),
        Parameter::slider("ao_power", "AO Power", (0.1, 8.0), (0.25, 4.0), 3, 1.0)
            .group(OCCLUSION)
            .help("AO's contrast: above 1 darkens partly open places more"),
        Parameter::slider("ao_direct", "AO on Direct", (0.0, 1.0), (0.0, 1.0), 3, 0.0)
            .group(OCCLUSION)
            .help("How much AO also darkens direct light, for contact shadows in dense foliage"),
        Parameter::slider("gi", "GI", (0.0, 1000.0), (0.0, 4.0), 3, 1.0)
            .group(BOUNCE)
            .help("Scales one bounce of light off nearby surfaces; 0 turns it off"),
        Parameter::integer("gi_samples", "GI Samples", 1, i64::from(MAX_GI_SAMPLES), 16)
            .group(BOUNCE)
            .help("Bounce rays a pixel traces, in a fixed pattern")
            .animatable(false),
        Parameter::length(
            "gi_distance",
            "GI Distance",
            (0.0, 1e6),
            Some((0.0, 20.0)),
            2,
            4.0,
        )
        .group(BOUNCE)
        .help("How far bounce rays look"),
        Parameter::integer(
            "volume_steps",
            "Volume Steps",
            1,
            i64::from(MAX_VOLUME_STEPS),
            24,
        )
        .group(VOLUMES)
        .help("Steps a ray takes through volumes, and as many through lit fog")
        .animatable(false),
        Parameter::length(
            "volume_max_distance",
            "Volume Max Distance",
            (0.0, 1e6),
            Some((0.0, 500.0)),
            2,
            100.0,
        )
        .group(VOLUMES)
        .help("How far from the camera volumes and lit fog are marched"),
        Parameter::new(
            "volume_shadow_rays",
            "Volume Shadow Rays",
            ValueKind::Toggle,
            true,
        )
        .group(VOLUMES)
        .help("Each step traces shadow rays: light shafts and a darker inside under cover")
        .animatable(false),
        Parameter::integer("volume_levels", "Volume Levels", 2, most, 12)
            .group(VOLUMES)
            .help("Levels of the opacity and light of volumes and fog"),
        Parameter::slider(
            "volume_dither",
            "Volume Dither",
            (0.0, 1.0),
            (0.0, 1.0),
            3,
            0.25,
        )
        .group(VOLUMES)
        .help("Dither on volumes and fog: below 1 only the bands' edges dither"),
    ]
    .into_iter()
    .fold(sections, Schema::parameter)
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
    let values = schema().fit(values);
    // Fitted: every value is there, of its kind.
    let number = |name: &str| values.number(name).unwrap_or(0.0) as f32;
    let integer = |name: &str| values.integer(name).unwrap_or(0);
    let count = |name: &str| integer(name).max(1) as u32;
    let key = values.choice("dither").unwrap_or("");
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
        assert_eq!(schema().check(), Ok(()));
        assert!(schema().parameters().all(|d| !d.group.is_empty()));
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
        let fixed: Vec<String> = (schema().parameters())
            .filter(|d| !d.animatable)
            .map(|d| d.name.to_string())
            .collect();
        assert_eq!(fixed, ["gi_samples", "volume_steps", "volume_shadow_rays"]);
    }
}
