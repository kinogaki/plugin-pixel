//! The Shade Ramp: the color each quantized light level takes, from the
//! darkest level (0, the shadow tint) to the brightest (1), cool shadows to
//! warm highlights. Its keys ride in the frame uniform; the shade pass
//! samples it as [`Ramp::at`] does, the SDK's ramp maths
//! (`kinogaki_render_api::ramp`) in each key's interpolation.

use kinogaki_render_api::ramp::{Interpolation, Key, Keys};

/// The most keys (`kinogaki_render_api::MAX_RAMP_STOPS`).
pub const STOPS: usize = kinogaki_render_api::MAX_RAMP_STOPS;

/// A color ramp in the GPU's layout: each key's color (rgb) and position
/// (w), ascending, and its interpolation's code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ramp {
    /// The keys; the first `count` hold values.
    pub stops: [[f32; 4]; STOPS],
    /// Each key's interpolation code (`Interpolation::code`).
    pub modes: [u32; STOPS],
    /// Keys in use (at least 1).
    pub count: u32,
}

impl Ramp {
    /// The ramp of `keys` (the first [`STOPS`]); a grey ramp when empty.
    pub fn new(keys: &[Key]) -> Ramp {
        if keys.is_empty() {
            return Ramp::grey();
        }
        let mut ramp = Ramp {
            stops: [[0.0; 4]; STOPS],
            modes: [0; STOPS],
            count: keys.len().min(STOPS) as u32,
        };
        for (at, k) in keys.iter().take(STOPS).enumerate() {
            let c = k.value.map(|v| v as f32);
            ramp.stops[at] = [c[0], c[1], c[2], k.position as f32];
            ramp.modes[at] = k.interpolation.code() as u32;
        }
        ramp
    }

    /// Black to white, linear: each level its own grey (the light's
    /// luminance).
    pub fn grey() -> Ramp {
        Ramp::new(&[
            Key::color(0.0, [0.0; 3], Interpolation::Linear),
            Key::color(1.0, [1.0; 3], Interpolation::Linear),
        ])
    }

    /// The color at level `t` (0 darkest, 1 brightest): between the keys
    /// around it as the first one's interpolation runs, the end keys'
    /// beyond them.
    pub fn at(&self, t: f32) -> [f32; 3] {
        [0, 1, 2].map(|c| self.evaluate_channel(f64::from(t), c) as f32)
    }
}

impl Keys for Ramp {
    fn count(&self) -> usize {
        self.count.max(1) as usize
    }
    fn position(&self, k: usize) -> f64 {
        f64::from(self.stops[k][3])
    }
    fn value(&self, k: usize, channel: usize) -> f64 {
        f64::from(self.stops[k][channel])
    }
    fn interpolation(&self, k: usize) -> Interpolation {
        Interpolation::from_code(f64::from(self.modes[k])).unwrap_or_default()
    }
}

/// The default ramp: deep blue-violet shadow, a cool middle, a neutral
/// light and a warm highlight (linear ACEScg), linear between them.
pub fn default_stops() -> Vec<Key> {
    let key = |p, rgb| Key::color(p, rgb, Interpolation::Linear);
    vec![
        key(0.0, [0.035, 0.03, 0.09]),
        key(0.34, [0.22, 0.23, 0.36]),
        key(0.67, [0.68, 0.66, 0.62]),
        key(1.0, [1.12, 0.98, 0.76]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinogaki_render_api::ramp;

    /// The default keys, each interpolated as `i`.
    fn keys_in(i: Interpolation) -> Vec<Key> {
        default_stops()
            .into_iter()
            .map(|k| Key {
                interpolation: i,
                ..k
            })
            .collect()
    }

    #[test]
    fn ramps_match_the_render_interface_in_every_interpolation() {
        for i in Interpolation::ALL {
            let keys = keys_in(i);
            let ours = Ramp::new(&keys);
            for t in [-0.5, 0.0, 0.1, 0.2, 0.34, 0.5, 0.6, 0.9, 1.0, 1.5] {
                let got = ours.at(t);
                let want = ramp::color_at(&keys, f64::from(t));
                for c in 0..3 {
                    assert!((got[c] - want[c] as f32).abs() < 1e-5, "{i:?} at {t}");
                }
            }
        }
        assert_eq!(Ramp::grey().at(0.25), [0.25; 3]);
        assert_eq!(Ramp::new(&[]), Ramp::grey());
        // Constant holds a key's color up to the next.
        let constant = Ramp::new(&keys_in(Interpolation::Constant));
        assert_eq!(constant.at(0.5), constant.at(0.34));
    }
}
