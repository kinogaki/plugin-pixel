//! The Shade Ramp: the color each quantized light level takes, from the
//! darkest level (0, the shadow tint) to the brightest (1), cool shadows to
//! warm highlights. Its stops ride in the frame uniform; the shade pass
//! samples it as [`Ramp::at`] does.

use kinogaki_render_api::ColorStop;

/// The most stops (`kinogaki_render_api::MAX_RAMP_STOPS`).
pub const STOPS: usize = kinogaki_render_api::MAX_RAMP_STOPS;

/// A color ramp in the GPU's layout: each stop's color (rgb) and position
/// (w), ascending.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ramp {
    /// The stops; the first `count` hold values.
    pub stops: [[f32; 4]; STOPS],
    /// Stops in use (at least 1).
    pub count: u32,
}

impl Ramp {
    /// The ramp of `stops` (the first [`STOPS`]); a grey ramp when empty.
    pub fn new(stops: &[ColorStop]) -> Ramp {
        if stops.is_empty() {
            return Ramp::grey();
        }
        let mut ramp = Ramp {
            stops: [[0.0; 4]; STOPS],
            count: stops.len().min(STOPS) as u32,
        };
        for (out, s) in ramp.stops.iter_mut().zip(stops) {
            let c = s.color.map(|v| v as f32);
            *out = [c[0], c[1], c[2], s.position as f32];
        }
        ramp
    }

    /// Black to white: each level its own grey (the light's luminance).
    pub fn grey() -> Ramp {
        Ramp::new(&[ColorStop::new(0.0, [0.0; 3]), ColorStop::new(1.0, [1.0; 3])])
    }

    /// The color at level `t` (0 darkest, 1 brightest): linear between the
    /// stops around it, the end stops' beyond them.
    pub fn at(&self, t: f32) -> [f32; 3] {
        let stops = &self.stops[..self.count.max(1) as usize];
        let rgb = |s: &[f32; 4]| [s[0], s[1], s[2]];
        if t.is_nan() || t <= stops[0][3] {
            return rgb(&stops[0]);
        }
        for pair in stops.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if t <= b[3] {
                let span = b[3] - a[3];
                let u = if span > 0.0 { (t - a[3]) / span } else { 1.0 };
                return [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * u);
            }
        }
        rgb(&stops[stops.len() - 1])
    }
}

/// The default ramp: deep blue-violet shadow, a cool middle, a neutral
/// light and a warm highlight (linear ACEScg).
pub fn default_stops() -> Vec<ColorStop> {
    vec![
        ColorStop::new(0.0, [0.035, 0.03, 0.09]),
        ColorStop::new(0.34, [0.22, 0.23, 0.36]),
        ColorStop::new(0.67, [0.68, 0.66, 0.62]),
        ColorStop::new(1.0, [1.12, 0.98, 0.76]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramps_blend_between_stops_and_match_the_render_interface() {
        let stops = default_stops();
        let ramp = Ramp::new(&stops);
        for t in [-0.5, 0.0, 0.1, 0.34, 0.5, 0.9, 1.0, 1.5] {
            let ours = ramp.at(t);
            let theirs = kinogaki_render_api::ramp_color(&stops, f64::from(t));
            for i in 0..3 {
                assert!((ours[i] - theirs[i] as f32).abs() < 1e-6, "{t}");
            }
        }
        assert_eq!(Ramp::grey().at(0.25), [0.25; 3]);
        assert_eq!(Ramp::new(&[]), Ramp::grey());
    }
}
