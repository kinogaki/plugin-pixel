// The fixed cosine-weighted hemisphere pattern AO and GI rays share: the
// same Fibonacci directions for every pixel and frame, in a tangent frame
// built from the normal alone, so what they trace is a continuous function
// of the surface point and normal (still while the camera holds, smooth
// while it moves).

const GOLDEN_ANGLE: f32 = 2.39996322972865;

// Duff et al.'s branchless orthonormal basis around `n`.
fn tangent_frame(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let u = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, u, n);
}

// The k-th of `count` cosine-weighted directions, in tangent space.
fn hemisphere_direction(k: u32, count: u32) -> vec3<f32> {
    let u = (f32(k) + 0.5) / f32(count);
    let r = sqrt(u);
    let phi = f32(k) * GOLDEN_ANGLE;
    return vec3<f32>(r * cos(phi), r * sin(phi), sqrt(1.0 - u));
}
