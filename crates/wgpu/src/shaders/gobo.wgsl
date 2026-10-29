// The gobo's mask: value-noise fBm shaped by contrast. Mirrors the pixel
// crate's gobo.rs line for line.

fn gobo_hash(x: i32, y: i32, seed: u32) -> f32 {
    var h = (u32(x) * 0x8da6b343u) ^ (u32(y) * 0xd8163841u) ^ (seed * 0xcb1ab31fu);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return f32(h >> 8u) / 16777216.0;
}

fn value_noise(p: vec2<f32>, seed: u32) -> f32 {
    let f = floor(p);
    let i = vec2<i32>(f);
    let t = p - f;
    let s = t * t * (3.0 - 2.0 * t);
    let a = gobo_hash(i.x, i.y, seed);
    let b = gobo_hash(i.x + 1, i.y, seed);
    let c = gobo_hash(i.x, i.y + 1, seed);
    let d = gobo_hash(i.x + 1, i.y + 1, seed);
    let bottom = a + (b - a) * s.x;
    let top = c + (d - c) * s.x;
    return bottom + (top - bottom) * s.y;
}

fn fbm(p: vec2<f32>, seed: u32) -> f32 {
    var sum = 0.0;
    var weight = 0.5;
    var scale = 1.0;
    for (var octave = 0u; octave < 3u; octave++) {
        let shift = 17.0 * f32(octave);
        let q = vec2<f32>(p.x * scale + shift, p.y * scale - shift);
        sum += weight * value_noise(q, seed + octave);
        weight *= 0.5;
        scale *= 2.0;
    }
    return sum / 0.875;
}

fn gobo_mask(p: vec2<f32>, contrast: f32, seed: u32) -> f32 {
    let n = fbm(p, seed);
    let w = max(0.5 * (1.0 - clamp(contrast, 0.0, 1.0)), 0.01);
    let t = clamp((n - (0.5 - w)) / (2.0 * w), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}
