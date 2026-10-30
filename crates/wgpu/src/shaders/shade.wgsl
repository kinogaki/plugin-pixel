// The shade pass: direct + ambient + indirect + rim light quantized by its
// luminance to shade_levels levels, an ordered Bayer dither between
// adjacent levels (anchored to film pixels), each level its Shade Ramp
// color in the light's chromaticity, × albedo; then the volumes over it
// (the volume pass's L and 1 − T: L_surface · T + L, with 1 − T and the
// media's own light, L / (1 − T), through the Shade Ramp, on the volumes'
// own levels and dither); then, unless the fog is lit (and so among the
// volumes), height fog, its amount on the same levels.
// Mirrors the pixel crate's dither.rs and shading.rs.

@group(0) @binding(1) var albedo_tex: texture_2d<f32>;
@group(0) @binding(2) var direct_tex: texture_2d<f32>;
@group(0) @binding(3) var ambient_tex: texture_2d<f32>;
@group(0) @binding(4) var indirect_tex: texture_2d<f32>;
@group(0) @binding(5) var id_tex: texture_2d<u32>;
@group(0) @binding(6) var normal_tex: texture_2d<f32>;
@group(0) @binding(7) var depth_tex: texture_2d<f32>;
@group(0) @binding(8) var beauty_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var volume_tex: texture_2d<f32>;

// ACEScg luminance (dither::ACESCG_LUMINANCE).
const LUMINANCE: vec3<f32> = vec3<f32>(0.2722287, 0.6740818, 0.0536895);
// How far a background pixel's fog looks.
const FAR: f32 = 1e30;

fn threshold(p: vec2<u32>) -> f32 {
    let n = frame.view.z;
    if n == 0u {
        return 0.5;
    }
    let i = (p.y % n) * n + p.x % n;
    return frame.bayer[i / 4u][i % 4u];
}

fn quantize(value: f32, levels: f32, t: f32, strength: f32) -> f32 {
    let steps = levels - 1.0;
    let v = max(value, 0.0) * steps;
    let base = floor(v);
    let cut = 0.5 + (t - 0.5) * strength;
    return (base + select(0.0, 1.0, v - base > cut)) / steps;
}

// The Shade Ramp's keys: count, position, color and interpolation code.
fn ramp_count() -> u32 {
    return max(u32(frame.shade.z), 1u);
}
fn ramp_at(k: u32) -> f32 {
    return frame.ramp[k].w;
}
fn ramp_value(k: u32) -> vec3<f32> {
    return frame.ramp[k].rgb;
}
fn ramp_mode(k: u32) -> u32 {
    return frame.ramp_modes[k / 4u][k % 4u];
}

// Interpolation codes (kinogaki_render_api::ramp::Interpolation::code).
const CONSTANT: u32 = 0u;
const LINEAR: u32 = 1u;
const CATMULL_ROM: u32 = 2u;
const MONOTONE: u32 = 3u;
const B_SPLINE: u32 = 4u;

// Whether key p has a distinct key before it, or after it.
fn ramp_before(p: u32) -> bool {
    return p > 0u && ramp_at(p - 1u) < ramp_at(p);
}
fn ramp_after(p: u32) -> bool {
    return p + 1u < ramp_count() && ramp_at(p + 1u) > ramp_at(p);
}

// The slope of the line from key p to the next.
fn ramp_secant(p: u32) -> vec3<f32> {
    return (ramp_value(p + 1u) - ramp_value(p)) / (ramp_at(p + 1u) - ramp_at(p));
}

// Catmull-Rom's tangent at key p: along its neighbours, flat at an end.
fn smooth_slope(p: u32) -> vec3<f32> {
    if !ramp_before(p) || !ramp_after(p) {
        return vec3<f32>(0.0);
    }
    return (ramp_value(p + 1u) - ramp_value(p - 1u)) / (ramp_at(p + 1u) - ramp_at(p - 1u));
}

// Rust's signum: 1 for zero.
fn signum(x: vec3<f32>) -> vec3<f32> {
    return select(vec3<f32>(1.0), vec3<f32>(-1.0), x < vec3<f32>(0.0));
}

// An end key's tangent from the near and far secants (spans ns, fs), kept
// from overshooting.
fn end_slope(near: vec3<f32>, far: vec3<f32>, ns: f32, fs: f32) -> vec3<f32> {
    let slope = ((2.0 * ns + fs) * near - ns * far) / (ns + fs);
    let capped = select(slope, 3.0 * near,
        signum(near) != signum(far) & abs(slope) > 3.0 * abs(near));
    return select(capped, vec3<f32>(0.0), signum(slope) != signum(near));
}

// The monotone cubic's tangent at key p (Fritsch–Butland inside, a
// shape-preserving three-point estimate at the ends).
fn monotone_slope(p: u32) -> vec3<f32> {
    let b = ramp_before(p);
    let a = ramp_after(p);
    if b && a {
        let left = ramp_secant(p - 1u);
        let right = ramp_secant(p);
        let ls = ramp_at(p) - ramp_at(p - 1u);
        let rs = ramp_at(p + 1u) - ramp_at(p);
        let leading = 2.0 * rs + ls;
        let trailing = rs + 2.0 * ls;
        let slope = (leading + trailing) / (leading / left + trailing / right);
        return select(slope, vec3<f32>(0.0), left * right <= vec3<f32>(0.0));
    }
    if a {
        let near = ramp_secant(p);
        if !ramp_after(p + 1u) {
            return near;
        }
        return end_slope(near, ramp_secant(p + 1u),
            ramp_at(p + 1u) - ramp_at(p), ramp_at(p + 2u) - ramp_at(p + 1u));
    }
    if b {
        let near = ramp_secant(p - 1u);
        if !ramp_before(p - 1u) {
            return near;
        }
        return end_slope(near, ramp_secant(p - 2u),
            ramp_at(p) - ramp_at(p - 1u), ramp_at(p - 1u) - ramp_at(p - 2u));
    }
    return vec3<f32>(0.0);
}

// Settings::shade_ramp at level t (Ramp::at, the SDK's Keys::evaluate_channel
// on each channel).
fn ramp(t: f32) -> vec3<f32> {
    let last = ramp_count() - 1u;
    if t <= ramp_at(0u) {
        return ramp_value(0u);
    }
    if t >= ramp_at(last) {
        return ramp_value(last);
    }
    // The span [lo, lo + 1] holding t.
    var lo = 0u;
    for (var k = 1u; k < last; k++) {
        if ramp_at(k) <= t {
            lo = k;
        }
    }
    let span = ramp_at(lo + 1u) - ramp_at(lo);
    let u = (t - ramp_at(lo)) / span;
    let start = ramp_value(lo);
    let end = ramp_value(lo + 1u);
    let u2 = u * u;
    let u3 = u2 * u;
    switch ramp_mode(lo) {
        case CONSTANT: {
            return start;
        }
        case CATMULL_ROM, MONOTONE: {
            var s0 = smooth_slope(lo);
            var s1 = smooth_slope(lo + 1u);
            if ramp_mode(lo) == MONOTONE {
                s0 = monotone_slope(lo);
                s1 = monotone_slope(lo + 1u);
            }
            return (2.0 * u3 - 3.0 * u2 + 1.0) * start
                + (u3 - 2.0 * u2 + u) * s0 * span
                + (3.0 * u2 - 2.0 * u3) * end
                + (u3 - u2) * s1 * span;
        }
        case B_SPLINE: {
            // The ends are mirrored, so the curve meets the end keys.
            let a = select(2.0 * start - end, ramp_value(max(lo, 1u) - 1u), ramp_before(lo));
            let d = select(2.0 * end - start, ramp_value(min(lo + 2u, last)), ramp_after(lo + 1u));
            let v = 1.0 - u;
            return (v * v * v * a
                + (3.0 * u3 - 6.0 * u2 + 4.0) * start
                + (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) * end
                + u3 * d) / 6.0;
        }
        case LINEAR, default: {
            return start + (end - start) * u;
        }
    }
}

// shading::shade on `levels`, dithered at `strength`.
fn shade(light: vec3<f32>, t: f32, levels: f32, strength: f32) -> vec3<f32> {
    let y = dot(light, LUMINANCE);
    let q = quantize(y, levels, t, strength);
    let color = ramp(min(q, 1.0)) * max(q, 1.0);
    if y <= 1e-6 {
        return color;
    }
    return color * light / y;
}

// shading::fog.
fn fog(origin: vec3<f32>, direction: vec3<f32>, t: f32) -> f32 {
    let density = frame.fog_color.w;
    if density <= 0.0 {
        return 0.0;
    }
    let up = frame.world_up.xyz;
    let start = min(frame.fog.x, t);
    let entry = origin + direction * start;
    let length = t - start;
    let base = density * exp(-frame.fog.z * (dot(entry, up) - frame.fog.y));
    let k = frame.fog.z * dot(direction, up);
    var depth = base * length;
    if abs(k) > 1e-5 {
        depth = base * (1.0 - exp(-k * length)) / k;
    }
    return 1.0 - exp(-max(depth, 0.0));
}

// shading::Bands::opacity: a medium's opacity on the volumes' levels.
fn banded(opacity: f32, t: f32) -> f32 {
    return min(quantize(opacity, frame.bands.x, t, frame.bands.y), 1.0);
}

// shading::over: `behind` seen through the volumes at film pixel `p`.
fn over(behind: vec3<f32>, p: vec2<u32>, t: f32) -> vec4<f32> {
    let v = textureLoad(volume_tex, p, 0);
    if v.a <= 1e-4 {
        return vec4<f32>(behind, 0.0);
    }
    let opacity = banded(v.a, t);
    return vec4<f32>(mix(behind, shade(v.rgb / v.a, t, frame.bands.x, frame.bands.y), opacity), opacity);
}

// Height fog's share along the ray to `reach`, on the volumes' levels;
// none when the fog is lit (the volume pass marched it).
fn haze(origin: vec3<f32>, d: vec3<f32>, reach: f32, t: f32) -> f32 {
    if frame.media.z == 1u {
        return 0.0;
    }
    return banded(fog(origin, d, reach), t);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.xy;
    if !inside_film(p) {
        return;
    }
    let t = threshold(p);
    let origin = ray_origin(p);
    let d = view_direction(p);
    if textureLoad(id_tex, p, 0).x == MISS {
        // The background (the domes, above or below the horizon, when the
        // camera sees them) through the volumes, fogged as far as the view
        // looks.
        let dome = select(frame.ground.rgb, frame.ambient.rgb, dot(d, frame.world_up.xyz) >= 0.0);
        let sky = dome * frame.ground.w;
        let through = over(sky, p, t);
        let f = haze(origin, d, FAR, t);
        let seen = mix(through.rgb, frame.fog_color.rgb, f);
        textureStore(beauty_out, p, vec4<f32>(seen, max(max(frame.ground.w, through.a), f)));
        return;
    }
    let direct = textureLoad(direct_tex, p, 0).rgb;
    var light = direct + textureLoad(ambient_tex, p, 0).rgb + textureLoad(indirect_tex, p, 0).rgb;
    // The rim lights: a Fresnel-like edge (shading::rim), the Lit ones
    // only where direct light reaches.
    let n = textureLoad(normal_tex, p, 0).xyz;
    let edge = pow(1.0 - clamp(dot(n, -d), 0.0, 1.0), 4.0);
    let lit = select(0.0, 1.0, dot(direct, LUMINANCE) > 1e-4);
    light += (frame.rim_all.rgb + frame.rim.rgb * lit) * edge;
    let albedo = textureLoad(albedo_tex, p, 0).rgb;
    let shaded = over(albedo * shade(light, t, frame.shade.x, frame.shade.y), p, t).rgb;
    // Fog to the surface, along the view.
    let depth = textureLoad(depth_tex, p, 0).x;
    let reach = length(world_position(p, depth) - origin);
    let f = haze(origin, d, reach, t);
    textureStore(beauty_out, p, vec4<f32>(mix(shaded, frame.fog_color.rgb, f), 1.0));
}
