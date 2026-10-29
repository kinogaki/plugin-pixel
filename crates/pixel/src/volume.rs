//! The Pixel Volume: a participating medium filling a closed mesh, and the
//! CPU definitions of its density field and phase function, which
//! `shaders/medium.wgsl` mirrors line for line.
//!
//! - **Density** is the extinction coefficient: the share of light a unit
//!   of distance takes out, so a ray through length `d` of it keeps
//!   `e^(−density·d)` (Beer–Lambert). Grey: every channel the same.
//! - **Scatter Color** is the single-scattering albedo: of what is taken
//!   out, this share (per channel) is scattered; the rest is absorbed. White
//!   scatters everything (bright mist), black absorbs everything (smoke).
//! - **Anisotropy** is Henyey–Greenstein's g: above 0 light scatters
//!   forward (bright looking toward the light: god rays), below 0 back.
//! - **Emission** × **Emission Color** is light each unit of density gives
//!   off a unit of distance.
//! - **Noise** shapes the density with value-noise fBm in world space,
//!   and **Edge Falloff** fades it in from the mesh's bounding box faces
//!   (in the mesh's own axes; exact for a box).

/// The most octaves Noise Octaves takes.
pub const MAX_OCTAVES: u32 = 8;

/// A Pixel Volume's inputs, as a frame renders them.
#[derive(Clone, Debug, PartialEq)]
pub struct PixelVolume {
    /// Extinction a unit of distance.
    pub density: f32,
    /// The single-scattering albedo, linear ACEScg.
    pub scatter_color: [f32; 3],
    /// Henyey–Greenstein g, in −0.9..=0.9.
    pub anisotropy: f32,
    /// Light given off a unit of density and distance.
    pub emission: f32,
    /// The emitted light's color, linear ACEScg.
    pub emission_color: [f32; 3],
    /// How much the noise shapes the density (0: even).
    pub noise_amount: f32,
    /// Noise cycles a scene unit.
    pub noise_frequency: f32,
    /// Noise octaves, each twice as fine and half as strong.
    pub noise_octaves: u32,
    /// Moves the noise, scene units.
    pub noise_offset: [f32; 3],
    /// The noise's seed.
    pub noise_seed: u32,
    /// How far in from the bounds the density fades in, scene units (0:
    /// a hard edge).
    pub edge_falloff: f32,
    /// Whether it dims the lights' light passing through it: shadows on
    /// surfaces and on itself (costly).
    pub shadows: bool,
}

impl Default for PixelVolume {
    /// A thin grey mist scattering a little forward.
    fn default() -> PixelVolume {
        PixelVolume {
            density: 0.5,
            scatter_color: [0.8; 3],
            anisotropy: 0.2,
            emission: 0.0,
            emission_color: [1.0; 3],
            noise_amount: 0.0,
            noise_frequency: 0.5,
            noise_octaves: 3,
            noise_offset: [0.0; 3],
            noise_seed: 0,
            edge_falloff: 0.0,
            shadows: false,
        }
    }
}

impl PixelVolume {
    /// The inputs in their valid ranges.
    pub fn clamped(&self) -> PixelVolume {
        let color = |c: [f32; 3]| c.map(|v| if v.is_finite() { v.max(0.0) } else { 0.0 });
        let positive = |v: f32| if v.is_finite() { v.max(0.0) } else { 0.0 };
        PixelVolume {
            density: positive(self.density),
            scatter_color: color(self.scatter_color).map(|v| v.min(1.0)),
            anisotropy: if self.anisotropy.is_finite() {
                self.anisotropy.clamp(-0.9, 0.9)
            } else {
                0.0
            },
            emission: positive(self.emission),
            emission_color: color(self.emission_color),
            noise_amount: positive(self.noise_amount).min(1.0),
            noise_frequency: positive(self.noise_frequency),
            noise_octaves: self.noise_octaves.clamp(1, MAX_OCTAVES),
            edge_falloff: positive(self.edge_falloff),
            ..self.clone()
        }
    }
}

/// Henyey–Greenstein's phase function for `g` at the cosine `cos_theta`
/// between the light's travel and the scattered direction: a density over
/// the sphere (it integrates to 1).
pub fn phase(g: f32, cos_theta: f32) -> f32 {
    let g2 = g * g;
    let d = (1.0 + g2 - 2.0 * g * cos_theta).max(1e-6);
    (1.0 - g2) / (4.0 * std::f32::consts::PI * d * d.sqrt())
}

/// A hash of a 3-D lattice point and a seed to [0, 1).
pub fn hash3(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ (z as u32).wrapping_mul(0xcb1a_b31f)
        ^ seed.wrapping_mul(0x9e37_79b9);
    h = (h ^ (h >> 16)).wrapping_mul(0x7feb_352d);
    h = (h ^ (h >> 15)).wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    (h >> 8) as f32 / 16_777_216.0
}

/// Value noise at `p`: the lattice's hashes, blended by smoothstep.
pub fn value_noise3(p: [f32; 3], seed: u32) -> f32 {
    let f = p.map(f32::floor);
    let i = f.map(|v| v as i32);
    let s = [0, 1, 2].map(|k| {
        let t = p[k] - f[k];
        t * t * (3.0 - 2.0 * t)
    });
    let at = |dx: i32, dy: i32, dz: i32| hash3(i[0] + dx, i[1] + dy, i[2] + dz, seed);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(at(0, 0, 0), at(1, 0, 0), s[0]);
    let x10 = lerp(at(0, 1, 0), at(1, 1, 0), s[0]);
    let x01 = lerp(at(0, 0, 1), at(1, 0, 1), s[0]);
    let x11 = lerp(at(0, 1, 1), at(1, 1, 1), s[0]);
    lerp(lerp(x00, x10, s[1]), lerp(x01, x11, s[1]), s[2])
}

/// `octaves` of value noise, each twice as fine and half as strong, in
/// [0, 1].
pub fn fbm3(p: [f32; 3], octaves: u32, seed: u32) -> f32 {
    let (mut sum, mut total) = (0.0, 0.0);
    let (mut weight, mut scale) = (0.5, 1.0);
    for octave in 0..octaves.clamp(1, MAX_OCTAVES) {
        // Each octave is shifted so lattice points do not line up.
        let shift = 17.0 * octave as f32;
        let q = [
            p[0] * scale + shift,
            p[1] * scale - shift,
            p[2] * scale + shift,
        ];
        sum += weight * value_noise3(q, seed.wrapping_add(octave));
        total += weight;
        weight *= 0.5;
        scale *= 2.0;
    }
    sum / total
}

/// The noise's share of the density at world point `x`: `mix(1, shape,
/// amount)`, where the shape is the fBm pushed toward 0 and 1
/// (smoothstep from 0.3 to 0.7), so a full amount leaves clear gaps
/// between banks.
pub fn noise_share(v: &PixelVolume, x: [f32; 3]) -> f32 {
    if v.noise_amount <= 0.0 {
        return 1.0;
    }
    let p = [0, 1, 2].map(|k| x[k] * v.noise_frequency + v.noise_offset[k]);
    let n = fbm3(p, v.noise_octaves, v.noise_seed);
    let t = ((n - 0.3) / 0.4).clamp(0.0, 1.0);
    1.0 + (t * t * (3.0 - 2.0 * t) - 1.0) * v.noise_amount
}

/// The edge falloff's share at `inside`, the distance in from the nearest
/// bounding box face (scene units): smoothstep over `falloff`.
pub fn edge_share(inside: f32, falloff: f32) -> f32 {
    if falloff <= 0.0 {
        return 1.0;
    }
    let t = (inside / falloff).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_integrates_to_one_and_leans_forward() {
        for g in [-0.6, 0.0, 0.3, 0.8] {
            // ∫ p dω = 2π ∫ p(cos) dcos.
            let n = 20_000;
            let sum: f32 = (0..n)
                .map(|i| {
                    let c = -1.0 + 2.0 * (i as f32 + 0.5) / n as f32;
                    phase(g, c) * 2.0 / n as f32
                })
                .sum::<f32>()
                * 2.0
                * std::f32::consts::PI;
            assert!((sum - 1.0).abs() < 2e-3, "g={g}: {sum}");
        }
        assert!(phase(0.5, 1.0) > phase(0.5, -1.0));
        let iso = 1.0 / (4.0 * std::f32::consts::PI);
        assert!((phase(0.0, 0.3) - iso).abs() < 1e-7);
    }

    #[test]
    fn noise_stays_in_range_is_continuous_and_seeded() {
        for i in 0..400 {
            let p = [
                i as f32 * 0.137,
                (i % 17) as f32 * 0.29 - 2.0,
                i as f32 * -0.071,
            ];
            let n = fbm3(p, 4, 7);
            assert!((0.0..=1.0).contains(&n), "{n}");
            assert!((n - fbm3([p[0] + 1e-3, p[1], p[2]], 4, 7)).abs() < 0.02);
        }
        let differ = (0..400)
            .filter(|&i| {
                let p = [i as f32 * 0.31, 0.5, i as f32 * 0.17];
                (fbm3(p, 3, 1) - fbm3(p, 3, 2)).abs() > 0.1
            })
            .count();
        assert!(differ > 100, "{differ}");
    }

    #[test]
    fn noise_and_edges_shape_the_density() {
        let even = PixelVolume::default();
        assert_eq!(noise_share(&even, [1.0, 2.0, 3.0]), 1.0);
        let banks = PixelVolume {
            noise_amount: 1.0,
            ..PixelVolume::default()
        };
        let shares: Vec<f32> = (0..2000)
            .map(|i| noise_share(&banks, [i as f32 * 0.21, 0.3, i as f32 * 0.13]))
            .collect();
        assert!(shares.iter().any(|&s| s < 0.01) && shares.iter().any(|&s| s > 0.99));
        assert_eq!(edge_share(0.5, 0.0), 1.0);
        assert_eq!(edge_share(0.0, 1.0), 0.0);
        assert_eq!(edge_share(2.0, 1.0), 1.0);
        assert_eq!(edge_share(0.5, 1.0), 0.5);
    }

    #[test]
    fn inputs_clamp_to_their_ranges() {
        let wild = PixelVolume {
            density: -1.0,
            scatter_color: [2.0, -1.0, f32::NAN],
            anisotropy: 3.0,
            noise_octaves: 99,
            ..PixelVolume::default()
        }
        .clamped();
        assert_eq!(wild.density, 0.0);
        assert_eq!(wild.scatter_color, [1.0, 0.0, 0.0]);
        assert_eq!(wild.anisotropy, 0.9);
        assert_eq!(wild.noise_octaves, MAX_OCTAVES);
    }
}
