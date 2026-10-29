//! Shade quantization and the ordered (Bayer) dither between levels.
//!
//! These are the CPU definitions: the shade pass reads the same Bayer table
//! (uploaded from [`thresholds`]) and mirrors [`quantize`] line for line in
//! `shaders/shade.wgsl` ([`crate::shading`] has the rest of that pass).

use crate::settings::Dither;

/// ACEScg (AP1) to luminance Y: the second row of AP1 → XYZ.
pub const ACESCG_LUMINANCE: [f32; 3] = [0.272_228_7, 0.674_081_8, 0.053_689_5];

/// The `n`×`n` Bayer matrix (`n` a power of two, at least 1), row-major,
/// by the recursion `M₂ₙ = [[4M, 4M+2], [4M+3, 4M+1]]`.
pub fn bayer(n: u32) -> Vec<u32> {
    assert!(
        n.is_power_of_two(),
        "a Bayer matrix's side is a power of two"
    );
    let mut m = vec![0u32];
    let mut side = 1usize;
    while side < n as usize {
        let next = side * 2;
        let mut out = vec![0u32; next * next];
        for y in 0..side {
            for x in 0..side {
                let v = 4 * m[y * side + x];
                out[y * next + x] = v;
                out[y * next + x + side] = v + 2;
                out[(y + side) * next + x] = v + 3;
                out[(y + side) * next + x + side] = v + 1;
            }
        }
        m = out;
        side = next;
    }
    m
}

/// The dither's thresholds in (0, 1), row-major, `(m + ½) / n²` each; none
/// when off.
pub fn thresholds(dither: Dither) -> Vec<f32> {
    let n = dither.size();
    if n == 0 {
        return Vec::new();
    }
    let cells = (n * n) as f32;
    bayer(n)
        .into_iter()
        .map(|m| (m as f32 + 0.5) / cells)
        .collect()
}

/// The threshold of film pixel (`x`, `y`): the matrix tiles the film from
/// its top-left pixel. One half when off (plain rounding).
pub fn threshold_at(table: &[f32], size: u32, x: u32, y: u32) -> f32 {
    if size == 0 {
        return 0.5;
    }
    table[((y % size) * size + x % size) as usize]
}

/// `value` on `levels` evenly spaced levels (0 and 1 among them; values
/// above 1 continue in the same steps). The step a value falls in rises
/// when its fraction passes `½ + (threshold − ½) × strength`.
pub fn quantize(value: f32, levels: u32, threshold: f32, strength: f32) -> f32 {
    let steps = (levels.max(2) - 1) as f32;
    let t = value.max(0.0) * steps;
    let base = t.floor();
    let cut = 0.5 + (threshold - 0.5) * strength;
    let up = if t - base > cut { 1.0 } else { 0.0 };
    (base + up) / steps
}

/// ACEScg luminance.
pub fn luminance(c: [f32; 3]) -> f32 {
    c[0] * ACESCG_LUMINANCE[0] + c[1] * ACESCG_LUMINANCE[1] + c[2] * ACESCG_LUMINANCE[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayer_known_matrices() {
        assert_eq!(bayer(1), vec![0]);
        assert_eq!(bayer(2), vec![0, 2, 3, 1]);
        assert_eq!(
            bayer(4),
            vec![0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
        );
    }

    #[test]
    fn bayer_is_a_permutation_of_its_cells() {
        for n in [2u32, 4, 8] {
            let mut m = bayer(n);
            m.sort_unstable();
            assert_eq!(m, (0..n * n).collect::<Vec<_>>());
        }
    }

    #[test]
    fn bayer_spreads_evenly() {
        // Every 2×2 block of a Bayer matrix holds one value from each
        // quarter of the range: the ordered dither's even spread.
        for n in [4u32, 8] {
            let m = bayer(n);
            let quarter = n * n / 4;
            for by in (0..n).step_by(2) {
                for bx in (0..n).step_by(2) {
                    let mut q: Vec<u32> = [(0, 0), (1, 0), (0, 1), (1, 1)]
                        .iter()
                        .map(|(dx, dy)| m[((by + dy) * n + bx + dx) as usize] / quarter)
                        .collect();
                    q.sort_unstable();
                    assert_eq!(q, vec![0, 1, 2, 3], "n={n} block {bx},{by}");
                }
            }
        }
    }

    #[test]
    fn thresholds_are_centred_and_tile() {
        let t = thresholds(Dither::Bayer4);
        assert_eq!(t.len(), 16);
        let mean: f32 = t.iter().sum::<f32>() / 16.0;
        assert!((mean - 0.5).abs() < 1e-6);
        assert!(t.iter().all(|&v| v > 0.0 && v < 1.0));
        assert_eq!(threshold_at(&t, 4, 1, 0), threshold_at(&t, 4, 5, 8));
        assert!(thresholds(Dither::Off).is_empty());
        assert_eq!(threshold_at(&[], 0, 3, 7), 0.5);
    }

    #[test]
    fn quantize_rounds_without_dither() {
        // Four levels: 0, ⅓, ⅔, 1.
        assert_eq!(quantize(0.0, 4, 0.5, 0.0), 0.0);
        assert_eq!(quantize(0.16, 4, 0.5, 0.0), 0.0);
        assert_eq!(quantize(0.17, 4, 0.5, 0.0), 1.0 / 3.0);
        assert_eq!(quantize(0.5, 4, 0.5, 0.0), 1.0 / 3.0);
        assert_eq!(quantize(0.52, 4, 0.5, 0.0), 2.0 / 3.0);
        assert_eq!(quantize(1.0, 4, 0.5, 0.0), 1.0);
        assert_eq!(quantize(-1.0, 4, 0.5, 0.0), 0.0);
        // Above 1 the steps go on.
        assert_eq!(quantize(1.34, 4, 0.5, 0.0), 4.0 / 3.0);
    }

    #[test]
    fn quantize_keeps_exact_levels() {
        for levels in 2..9 {
            for k in 0..levels {
                let v = k as f32 / (levels - 1) as f32;
                for th in thresholds(Dither::Bayer8) {
                    assert_eq!(quantize(v, levels, th, 1.0), v);
                }
            }
        }
    }

    #[test]
    fn dither_mean_matches_the_value() {
        // Over one tile of thresholds, full-strength dither reproduces the
        // value to within one cell of the step.
        let table = thresholds(Dither::Bayer8);
        for i in 0..=100 {
            let v = i as f32 / 100.0;
            let mean: f32 = table.iter().map(|&t| quantize(v, 4, t, 1.0)).sum::<f32>() / 64.0;
            assert!(
                (mean - v).abs() <= 1.0 / 3.0 / 64.0 + 1e-6,
                "v={v} mean={mean}"
            );
        }
    }

    #[test]
    fn dither_strength_scales_the_spread() {
        // At strength ½ the thresholds span ¼..¾ of the step.
        assert_eq!(quantize(0.2 / 3.0, 4, 0.0, 0.5), 0.0);
        assert_eq!(quantize(0.3 / 3.0, 4, 0.0, 0.5), 1.0 / 3.0);
    }

    #[test]
    fn luminance_of_white_is_one() {
        assert!((luminance([1.0; 3]) - 1.0).abs() < 1e-6);
    }
}
