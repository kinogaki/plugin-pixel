//! The gobo (cookie): a mask of dappled light on Distant and Spot lights,
//! as if through leaves out of frame.
//!
//! Pixel adds it to the standard distant and spot lights as an applied
//! schema ([`schema`]): a toggle and the pattern's settings, decoded from a
//! light's values when on ([`Gobo::decode`]).
//!
//! The mask is value-noise fBm (three octaves) on the light's plane, in
//! gobo units (scene units over the gobo's scale), shaped by its contrast:
//! a smoothstep about one half, as wide as `1 − contrast`, so 0 is a soft
//! mottle and 1 hard light and shade. These are the CPU definitions;
//! `shaders/gobo.wgsl` mirrors them line for line.

use std::sync::Arc;

use kinogaki_render_api::settings::set_value;
use kinogaki_render_api::standard::{self, slider};
use kinogaki_render_api::{
    AppliedSchema, Declared, SettingDescriptor, SettingGroup, SettingKind, SettingValue, Unit,
};

/// The gobo schema's id.
pub const ID: &str = "kinogaki.pixel.gobo";

/// The group its settings sit in.
const GROUP: &str = "light-gobo";

/// Its parameters' names.
mod key {
    pub const GOBO: &str = "gobo";
    pub const SCALE: &str = "goboScale";
    pub const CONTRAST: &str = "goboContrast";
    pub const SEED: &str = "goboSeed";
    pub const OFFSET: &str = "goboOffset";
}

/// A light's cookie: a tiling noise that masks a Distant or Spot light's
/// direct light, projected along the light onto its own XY plane (a
/// spot's: at unit distance along -Z).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gobo {
    /// The pattern's features' size, in scene units on that plane.
    pub scale: f32,
    /// 0 a soft mottle, 1 hard-edged light and shade.
    pub contrast: f32,
    /// Picks the pattern.
    pub seed: u32,
    /// Slides the pattern across the plane, in scene units.
    pub offset: [f32; 2],
}

impl Default for Gobo {
    fn default() -> Gobo {
        Gobo {
            scale: 1.0,
            contrast: 0.5,
            seed: 0,
            offset: [0.0; 2],
        }
    }
}

impl Gobo {
    /// The gobo of `light` (a distant light or spot): none when its Gobo
    /// is off.
    pub fn decode(light: &Declared) -> Option<Gobo> {
        light.bool(key::GOBO).then(|| Gobo {
            scale: light.float(key::SCALE),
            contrast: light.float(key::CONTRAST),
            seed: light
                .value(key::SEED)
                .and_then(SettingValue::as_int)
                .map_or(0, |v| v.clamp(0, i64::from(u32::MAX)) as u32),
            offset: light
                .value(key::OFFSET)
                .and_then(SettingValue::as_vector2)
                .unwrap_or([0.0; 2])
                .map(|v| v as f32),
        })
    }

    /// `light` with `gobo` (none: Gobo off, the pattern's defaults).
    pub fn apply(gobo: Option<Gobo>, mut light: Declared) -> Declared {
        let g = gobo.unwrap_or_default();
        let float = |v: f32| SettingValue::Float(f64::from(v));
        let values = &mut light.values;
        set_value(values, key::GOBO, SettingValue::Bool(gobo.is_some()));
        set_value(values, key::SCALE, float(g.scale));
        set_value(values, key::CONTRAST, float(g.contrast));
        set_value(values, key::SEED, SettingValue::Int(i64::from(g.seed)));
        set_value(
            values,
            key::OFFSET,
            SettingValue::Vector2(g.offset.map(f64::from)),
        );
        light
    }
}

/// The gobo's parameters on the standard distant and spot lights: Gobo,
/// then Scale, Contrast, Seed and Offset, used with Gobo on.
pub fn schema() -> AppliedSchema {
    use key::*;
    let on = |d: SettingDescriptor| d.requires(GOBO, SettingValue::Bool(true), "Used with Gobo on");
    let row = |name, label, kind| on(SettingDescriptor::new(name, label, GROUP, kind));
    AppliedSchema {
        id: Arc::from(ID),
        applies_to: vec![
            Arc::from(standard::DISTANT_LIGHT),
            Arc::from(standard::SPOT_LIGHT),
        ],
        groups: vec![SettingGroup {
            open_with: Some(Arc::from(GOBO)),
            ..SettingGroup::new(GROUP, "Gobo", false)
        }],
        params: vec![
            SettingDescriptor::new(GOBO, "Gobo", GROUP, SettingKind::Bool { default: false }),
            row(
                SCALE,
                "Scale",
                SettingKind::Float {
                    default: 1.0,
                    min: 1e-6,
                    max: 1e12,
                    soft: (1e-6, 1e12),
                    decimals: 3,
                    unit: Unit::Length,
                    slider: false,
                    step: 0.01,
                },
            ),
            row(CONTRAST, "Contrast", slider(0.5, 0.0, 1.0, (0.0, 1.0), 2)),
            row(
                SEED,
                "Seed",
                SettingKind::Int {
                    default: 0,
                    min: 0,
                    max: i64::from(u32::MAX),
                },
            ),
            row(
                OFFSET,
                "Offset",
                SettingKind::Vector2 {
                    default: [0.0; 2],
                    min: -1e12,
                    max: 1e12,
                    unit: Unit::Length,
                    step: 0.01,
                },
            ),
        ],
    }
}

/// A hash of a lattice point and a seed to [0, 1).
pub fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ seed.wrapping_mul(0xcb1a_b31f);
    h = (h ^ (h >> 16)).wrapping_mul(0x7feb_352d);
    h = (h ^ (h >> 15)).wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    (h >> 8) as f32 / 16_777_216.0
}

/// Value noise at `p`: the lattice's hashes, blended by smoothstep.
pub fn value_noise(p: [f32; 2], seed: u32) -> f32 {
    let (fx, fy) = (p[0].floor(), p[1].floor());
    let (ix, iy) = (fx as i32, fy as i32);
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (u, v) = (s(p[0] - fx), s(p[1] - fy));
    let a = hash(ix, iy, seed);
    let b = hash(ix + 1, iy, seed);
    let c = hash(ix, iy + 1, seed);
    let d = hash(ix + 1, iy + 1, seed);
    let bottom = a + (b - a) * u;
    let top = c + (d - c) * u;
    bottom + (top - bottom) * v
}

/// Three octaves of value noise, each twice as fine and half as strong,
/// in [0, 1].
pub fn fbm(p: [f32; 2], seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut weight = 0.5;
    let mut scale = 1.0;
    for octave in 0..3u32 {
        // Each octave is shifted so lattice points do not line up.
        let shift = 17.0 * octave as f32;
        let q = [p[0] * scale + shift, p[1] * scale - shift];
        sum += weight * value_noise(q, seed.wrapping_add(octave));
        weight *= 0.5;
        scale *= 2.0;
    }
    sum / 0.875
}

/// The mask at `p` (gobo units) for `contrast` and `seed`, in [0, 1].
pub fn mask(p: [f32; 2], contrast: f32, seed: u32) -> f32 {
    let n = fbm(p, seed);
    let w = (0.5 * (1.0 - contrast.clamp(0.0, 1.0))).max(0.01);
    let t = ((n - (0.5 - w)) / (2.0 * w)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples() -> impl Iterator<Item = [f32; 2]> {
        (0..64).flat_map(|i| (0..64).map(move |j| [i as f32 * 0.173, j as f32 * 0.131 - 3.0]))
    }

    #[test]
    fn the_noise_stays_in_range_and_is_continuous() {
        for p in samples() {
            let n = fbm(p, 3);
            assert!((0.0..=1.0).contains(&n), "{n}");
            let near = fbm([p[0] + 1e-3, p[1]], 3);
            assert!((n - near).abs() < 0.02, "a step of 1e-3 moves it little");
        }
    }

    #[test]
    fn contrast_hardens_the_mask_and_seeds_change_it() {
        let share = |contrast| {
            let (mut hard, mut all) = (0, 0);
            for p in samples() {
                let m = mask(p, contrast, 1);
                all += 1;
                hard += usize::from(!(0.02..=0.98).contains(&m));
            }
            hard as f32 / all as f32
        };
        let (soft, hard) = (share(0.0), share(1.0));
        assert!(soft < 0.1 && hard > 0.9, "{soft} {hard}");
        let differ = samples()
            .filter(|&p| (mask(p, 0.5, 1) - mask(p, 0.5, 2)).abs() > 0.1)
            .count();
        assert!(differ > 500, "{differ}");
        // Both light and shade.
        let lit = samples().filter(|&p| mask(p, 1.0, 1) > 0.5).count();
        assert!((1000..3100).contains(&lit), "{lit}");
    }

    #[test]
    fn a_gobo_decodes_from_a_light_only_when_on() {
        let spot = standard::Light {
            shape: standard::Shape::Spot {
                radius: 0.1,
                cone_angle: 30.0,
                cone_softness: 0.2,
            },
            ..standard::Light::default()
        };
        let dapple = Gobo {
            scale: 2.0,
            contrast: 0.75,
            seed: 9,
            offset: [1.0, -4.0],
        };
        let on = Gobo::apply(Some(dapple), spot.declared());
        assert_eq!(Gobo::decode(&on), Some(dapple));
        assert_eq!(Gobo::decode(&Gobo::apply(None, spot.declared())), None);
        // The schema's defaults are the gobo's, off.
        let s = schema();
        let mut defaults = spot.declared();
        defaults
            .values
            .extend(kinogaki_render_api::settings::defaults(&s.params));
        assert_eq!(Gobo::decode(&defaults), None);
        set_value(&mut defaults.values, key::GOBO, SettingValue::Bool(true));
        assert_eq!(Gobo::decode(&defaults), Some(Gobo::default()));
        assert!(s.applies(standard::DISTANT_LIGHT) && s.applies(standard::SPOT_LIGHT));
        assert!(!s.applies(standard::AREA_LIGHT));
    }
}
