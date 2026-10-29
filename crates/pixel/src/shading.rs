//! The lighting's and shade pass's CPU definitions, mirrored line for line
//! by the shaders: AO shaped by its strength and power
//! (`shaders/ambient.wgsl`), light quantized through the Shade Ramp, the
//! rim light added before it, the volumes composed over it ([`over`]),
//! and fog mixed in after it, the media's opacity and light and the fog's
//! amount on their own [`Bands`] so mist posterizes into soft bands
//! (`shaders/shade.wgsl`).

use crate::dither::{luminance, quantize};
use crate::settings::Ramp;

/// Light (ACEScg, off a white surface) quantized by its luminance, the
/// level's ramp color in its chromaticity: `ramp(level) × light / Y`.
/// Levels past full light keep the top color, brighter. Unlit light takes
/// the darkest level's color, in white.
pub fn shade(light: [f32; 3], ramp: &Ramp, levels: u32, threshold: f32, strength: f32) -> [f32; 3] {
    let y = luminance(light);
    let q = quantize(y, levels, threshold, strength);
    let color = ramp.at(q.min(1.0)).map(|c| c * q.max(1.0));
    if y <= 1e-6 {
        return color;
    }
    [0, 1, 2].map(|i| color[i] * light[i] / y)
}

/// How the volumes and fog are quantized: their opacity and the media's
/// own light on `levels` levels of their own, dithered at `strength`
/// (below 1 only the bands' edges dither, so mist reads as soft bands).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bands {
    /// Levels between clear and opaque, and between unlit and full light
    /// (at least 2).
    pub levels: u32,
    /// The dither's strength, 0..=1.
    pub strength: f32,
}

impl Bands {
    /// The settings' Volume Levels and Volume Dither.
    pub fn of(settings: &crate::Settings) -> Bands {
        Bands {
            levels: settings.volume_levels,
            strength: settings.volume_dither,
        }
    }

    /// `opacity` (1 − T, or fog's share) on the bands, at most 1.
    pub fn opacity(self, opacity: f32, threshold: f32) -> f32 {
        quantize(opacity, self.levels, threshold, self.strength).min(1.0)
    }
}

/// `behind` seen through volumes whose light is `volume` (rgb, as the
/// volume output holds it) and opacity `opacity` (1 − T): `behind · T +
/// L`, written as `mix(behind, L / (1 − T), 1 − T)` so that the opacity
/// and the media's own light, `L / (1 − T)`, are quantized on their
/// [`Bands`], the light through the Shade Ramp as a surface's is.
pub fn over(
    behind: [f32; 3],
    (volume, opacity): ([f32; 3], f32),
    ramp: &Ramp,
    bands: Bands,
    threshold: f32,
) -> [f32; 3] {
    if opacity <= 1e-4 {
        return behind;
    }
    let q = bands.opacity(opacity, threshold);
    let own = shade(
        volume.map(|c| c / opacity),
        ramp,
        bands.levels,
        threshold,
        bands.strength,
    );
    [0, 1, 2].map(|i| behind[i] + (own[i] - behind[i]) * q)
}

/// AO as applied: the traced open share `traced` raised to `power`, mixed
/// toward 1 by `1 − strength`.
pub fn occlusion(traced: f32, strength: f32, power: f32) -> f32 {
    1.0 + (traced.powf(power) - 1.0) * strength
}

/// The rim light's share at a surface whose normal makes cosine `n_dot_v`
/// with the direction to the eye: (1 − n·v)⁴, a Fresnel-like edge.
pub fn rim(n_dot_v: f32) -> f32 {
    (1.0 - n_dot_v.clamp(0.0, 1.0)).powi(4)
}

/// Exponential height fog along a ray: the share of the fog's color, in
/// 0..=1, at distance `t` along `direction` (unit) from `origin`. The
/// density is the fog's density × e^(−falloff × the height above the fog's
/// height), from `start` along the ray on; falloff 0 is even distance fog.
pub fn fog(origin: [f32; 3], direction: [f32; 3], t: f32, f: &HeightFog) -> f32 {
    if f.density <= 0.0 {
        return 0.0;
    }
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let start = f.start.min(t);
    let from = [0, 1, 2].map(|i| origin[i] + direction[i] * start);
    let length = t - start;
    let base = f.density * (-f.falloff * (dot(from, f.up) - f.height)).exp();
    let k = f.falloff * dot(direction, f.up);
    let depth = if k.abs() > 1e-5 {
        base * (1.0 - (-k * length).exp()) / k
    } else {
        base * length
    };
    1.0 - (-depth.max(0.0)).exp()
}

/// The fog [`fog`] reads: a Fog prim's density field and the world's up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightFog {
    /// At the fog's height, a unit of distance.
    pub density: f32,
    /// From the camera.
    pub start: f32,
    /// Where `density` holds.
    pub height: f32,
    /// e-folds a unit up.
    pub falloff: f32,
    /// The world's up (unit).
    pub up: [f32; 3],
}

impl HeightFog {
    /// `fog`'s field about `up`.
    pub fn of(fog: &crate::Fog, up: [f32; 3]) -> HeightFog {
        HeightFog {
            density: fog.density,
            start: fog.start,
            height: fog.height,
            falloff: fog.falloff,
            up,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grey_ramp_keeps_chromaticity_and_levels() {
        let light = [0.6, 0.3, 0.1];
        let s = shade(light, &Ramp::grey(), 4, 0.5, 0.0);
        assert!((luminance(s) - 1.0 / 3.0).abs() < 1e-6);
        assert!((s[0] / s[1] - 2.0).abs() < 1e-5);
        assert_eq!(shade([0.0; 3], &Ramp::grey(), 4, 0.5, 1.0), [0.0; 3]);
    }

    #[test]
    fn levels_take_their_ramp_colors() {
        let ramp = Ramp::new(&crate::settings::ramp::default_stops());
        // Unlit: the darkest color (the shadow tint); full white light: the top.
        assert_eq!(shade([0.0; 3], &ramp, 4, 0.5, 1.0), ramp.at(0.0));
        let top = shade([1.0; 3], &ramp, 4, 0.5, 0.0);
        let want = ramp.at(1.0);
        for i in 0..3 {
            assert!((top[i] - want[i]).abs() < 1e-5);
        }
        // Past full light the top color brightens.
        let over = shade([2.0; 3], &ramp, 4, 0.5, 0.0);
        assert!((over[0] / top[0] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn fog_thickens_with_distance_and_thins_with_height() {
        let even = HeightFog {
            density: 0.5,
            start: 1.0,
            height: 0.0,
            falloff: 0.0,
            up: [0.0, 1.0, 0.0],
        };
        let level = [1.0, 0.0, 0.0];
        assert_eq!(fog([0.0; 3], level, 0.5, &even), 0.0, "before the start");
        let f = fog([0.0; 3], level, 3.0, &even);
        assert!((f - (1.0 - (-1.0f32).exp())).abs() < 1e-6);
        let none = HeightFog {
            density: 0.0,
            ..even
        };
        assert_eq!(fog([0.0; 3], level, 100.0, &none), 0.0);
        // Height fog: thinner high up; a ray going down matches the
        // closed form against a sum of short steps.
        let height = HeightFog {
            falloff: 1.0,
            start: 0.0,
            ..even
        };
        let low = fog([0.0, 0.0, 0.0], level, 2.0, &height);
        let high = fog([0.0, 3.0, 0.0], level, 2.0, &height);
        assert!(high < low * 0.1, "{low} {high}");
        let down = [0.0, -0.6, 0.8];
        let origin = [0.0, 4.0, 0.0];
        let steps = 20_000;
        let dt = 5.0 / steps as f32;
        let mut depth = 0.0;
        for i in 0..steps {
            let y = origin[1] + down[1] * (i as f32 + 0.5) * dt;
            depth += 0.5 * (-y).exp() * dt;
        }
        let want = 1.0 - (-depth).exp();
        let got = fog(origin, down, 5.0, &height);
        assert!((got - want).abs() < 1e-3, "{got} {want}");
    }

    #[test]
    fn volumes_compose_over_what_lies_behind() {
        let grey = Ramp::grey();
        // Unquantized (many levels): behind · T + L.
        let behind = [0.2, 0.4, 0.6];
        let (light, opacity) = ([0.3, 0.3, 0.3], 0.5);
        let fine = Bands {
            levels: 4097,
            strength: 0.0,
        };
        let got = over(behind, (light, opacity), &grey, fine, 0.5);
        for i in 0..3 {
            let want = behind[i] * (1.0 - opacity) + light[i];
            assert!((got[i] - want).abs() < 1e-3, "{got:?}");
        }
        let bands = Bands {
            levels: 4,
            strength: 1.0,
        };
        assert_eq!(over(behind, ([0.0; 3], 0.0), &grey, bands, 0.5), behind);
        // Thick mist hides what lies behind; its light takes a level.
        let three = Bands {
            levels: 3,
            strength: 0.0,
        };
        let thick = over(behind, ([0.5; 3], 1.0), &grey, three, 0.5);
        assert_eq!(thick, [0.5; 3]);
    }

    #[test]
    fn volume_opacity_takes_its_own_levels_and_dithers_only_at_band_edges() {
        let table = crate::dither::thresholds(crate::Dither::Bayer4);
        let bands = Bands {
            levels: 8,
            strength: 0.25,
        };
        let step = 1.0 / 7.0;
        for k in 0..=70 {
            let v = k as f32 / 70.0;
            let seen: Vec<f32> = table.iter().map(|&t| bands.opacity(v, t)).collect();
            // On the volume's own levels, whatever the shade levels.
            for q in &seen {
                let level = q / step;
                assert!((level - level.round()).abs() < 1e-5, "{v}: {q}");
            }
            // Within the middle of a band every pixel takes one level:
            // no checkerboard; only near a band's edge do they differ.
            let fraction = (v / step).fract();
            let spread = seen.iter().any(|&q| q != seen[0]);
            if (fraction - 0.5).abs() > 0.125 + 1e-4 {
                assert!(!spread, "{v} dithers in a band's middle");
            }
        }
        // At the edge itself half the tile takes each level.
        let edge = 3.5 * step;
        let up = table
            .iter()
            .filter(|&&t| bands.opacity(edge, t) > 3.5 * step);
        assert_eq!(up.count(), 8);
        // Full strength spreads the whole step, as the surfaces' dither.
        let full = Bands {
            strength: 1.0,
            ..bands
        };
        let near = 3.1 * step;
        assert!(table.iter().any(|&t| full.opacity(near, t) > near));
        assert!(table.iter().all(|&t| bands.opacity(near, t) < near));
    }

    #[test]
    fn volumes_compose_the_same_every_time() {
        let ramp = Ramp::new(&crate::settings::ramp::default_stops());
        let bands = Bands::of(&crate::Settings::default());
        let table = crate::dither::thresholds(crate::Dither::Bayer4);
        let compose = || -> Vec<[f32; 3]> {
            table
                .iter()
                .map(|&t| over([0.1, 0.2, 0.3], ([0.2, 0.25, 0.3], 0.37), &ramp, bands, t))
                .collect()
        };
        assert_eq!(compose(), compose());
    }

    #[test]
    fn occlusion_follows_its_strength_and_power() {
        assert_eq!(occlusion(0.25, 1.0, 1.0), 0.25, "as traced");
        assert_eq!(occlusion(0.25, 0.0, 3.0), 1.0, "ignored");
        assert_eq!(occlusion(0.25, 0.5, 1.0), 0.625);
        assert_eq!(occlusion(0.5, 1.0, 2.0), 0.25, "more contrast");
        assert_eq!(occlusion(1.0, 1.0, 4.0), 1.0, "open stays open");
    }

    #[test]
    fn rims_rise_toward_the_silhouette() {
        assert_eq!(rim(1.0), 0.0);
        assert_eq!(rim(0.0), 1.0);
        assert!(rim(0.3) > rim(0.6));
    }
}
