//! The fixed cosine-weighted hemisphere pattern AO and GI rays share,
//! mirrored line for line by `shaders/hemisphere.wgsl`: `count` Fibonacci
//! directions, the same for every pixel and frame, in a tangent frame built
//! from the normal alone. What they trace is then a continuous function of
//! the surface point and normal: still while the camera holds, smooth while
//! it moves.

/// The golden angle, radians.
const GOLDEN_ANGLE: f32 = 2.399_963_2;

/// Duff et al.'s branchless orthonormal basis around unit `n`: tangent,
/// bitangent, `n`.
pub fn tangent_frame(n: [f32; 3]) -> [[f32; 3]; 3] {
    let s = if n[2] >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n[2]);
    let b = n[0] * n[1] * a;
    [
        [1.0 + s * n[0] * n[0] * a, s * b, -s * n[0]],
        [b, s + n[1] * n[1] * a, -n[1]],
        n,
    ]
}

/// The `k`-th of `count` cosine-weighted directions, in tangent space.
pub fn direction(k: u32, count: u32) -> [f32; 3] {
    let u = (k as f32 + 0.5) / count as f32;
    let r = u.sqrt();
    let phi = k as f32 * GOLDEN_ANGLE;
    [r * phi.cos(), r * phi.sin(), (1.0 - u).sqrt()]
}

/// The `k`-th of `count` directions around unit `n`, in world space.
pub fn around(n: [f32; 3], k: u32, count: u32) -> [f32; 3] {
    let [t, b, n] = tangent_frame(n);
    let d = direction(k, count);
    [0, 1, 2].map(|i| t[i] * d[0] + b[i] * d[1] + n[i] * d[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn frames_are_orthonormal_and_directions_face_the_normal() {
        for n in [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.6, 0.0, 0.8],
            [0.0, -0.6, 0.8],
        ] {
            let [t, b, m] = tangent_frame(n);
            for (u, v) in [(t, b), (b, m), (t, m)] {
                assert!(dot(u, v).abs() < 1e-6);
            }
            assert!((dot(t, t) - 1.0).abs() < 1e-6 && (dot(b, b) - 1.0).abs() < 1e-6);
            for k in 0..16 {
                let d = around(n, k, 16);
                assert!(dot(d, n) > 0.0 && (dot(d, d) - 1.0).abs() < 1e-5);
            }
        }
        // Cosine-weighted: the mean cosine is 2/3.
        let mean = (0..1024).map(|k| direction(k, 1024)[2]).sum::<f32>() / 1024.0;
        assert!((mean - 2.0 / 3.0).abs() < 1e-3, "{mean}");
    }
}
