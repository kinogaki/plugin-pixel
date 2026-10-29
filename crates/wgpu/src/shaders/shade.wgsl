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

// Settings::shade_ramp at level t (Ramp::at).
fn ramp(t: f32) -> vec3<f32> {
    let count = max(u32(frame.shade.z), 1u);
    if t <= frame.ramp[0].w {
        return frame.ramp[0].rgb;
    }
    for (var i = 1u; i < count; i++) {
        let a = frame.ramp[i - 1u];
        let b = frame.ramp[i];
        if t <= b.w {
            let span = b.w - a.w;
            let u = select(1.0, (t - a.w) / span, span > 0.0);
            return mix(a.rgb, b.rgb, u);
        }
    }
    return frame.ramp[count - 1u].rgb;
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
