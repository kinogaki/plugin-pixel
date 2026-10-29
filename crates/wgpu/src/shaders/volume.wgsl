// The volume pass: each film pixel's camera ray marched through the
// volumes it crosses (medium.wgsl) and, when the fog is lit, through the
// fog, up to the first surface or volume_max_distance. It writes the light
// they scatter toward the camera and their opacity, premultiplied:
// rgb = L, a = 1 − T, for L_surface · T + L. The shade pass composes it.
//
// The crossings split the ray into stretches; a stretch inside a volume
// takes a share of volume_steps by its length, and with lit fog the
// stretches outside volumes take another volume_steps between them. Each
// step sits at a fixed jitter a film pixel (interleaved gradient noise), so
// nothing flickers while the camera moves. A step:
// - sums the media there: each volume's density (× its scatter color: what
//   scatters; × its emission) and the lit fog's (× Fog Color);
// - lights them: each light's arrival × the shadow ray to it (surfaces cut
//   the shafts; thin translucent ones let some through) × what shadowing
//   volumes let through × its gobo × each medium's phase; and the domes'
//   light (the mean of above and below the horizon) × whether one
//   ray toward the sky gets out (each step one direction of the fixed
//   hemisphere pattern around up);
// - integrates its stretch exactly for a constant medium: the light it
//   sends is T × source / σt × (1 − e^(−σt Δ)), and T keeps e^(−σt Δ).

@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var id_tex: texture_2d<u32>;
@group(0) @binding(3) var volume_out: texture_storage_2d<rgba16float, write>;
// Binding 4: the lights (light.wgsl).

// How far a step's sky ray looks.
const SKY: f32 = 1e30;

// The march so far.
struct March {
    light: vec3<f32>,
    transmittance: f32,
    // Steps taken: picks each step's sky direction.
    steps: u32,
}

// A fixed offset in [0, 1) for film pixel `p` (Jimenez's interleaved
// gradient noise): neighbours differ, frames do not.
fn jitter(p: vec2<u32>) -> f32 {
    let q = vec2<f32>(p);
    return fract(52.9829189 * fract(0.06711056 * q.x + 0.00583715 * q.y));
}

// One step at `t` along the ray (`x` there), `delta` long, inside the
// volumes of `mask` and, when `fog` is set, the lit fog.
fn march_step(m: ptr<function, March>, x: vec3<f32>, t: f32, delta: f32, d: vec3<f32>, mask: u32, fog: bool) {
    // What there is here: extinction, what scatters (by medium) and emits.
    var sigma = 0.0;
    var scatter = vec3<f32>(0.0);
    var emitted = vec3<f32>(0.0);
    var density: array<f32, 32>;
    var bits = mask;
    while bits != 0u {
        let v = firstTrailingBit(bits);
        bits &= bits - 1u;
        let s = medium_density(v, x);
        density[v] = s;
        sigma += s;
        scatter += s * media[v].scatter.rgb;
        emitted += s * media[v].emission.rgb;
    }
    var fog_sigma = 0.0;
    if fog {
        fog_sigma = fog_density(x, t);
        sigma += fog_sigma;
        scatter += fog_sigma * frame.fog_color.rgb;
    }
    if sigma <= 0.0 {
        return;
    }
    let shadows = frame.volume.z > 0.5;
    var lit = vec3<f32>(0.0);
    for (var i = 0u; i < frame.film.w; i++) {
        let l = lights[i];
        let a = arrival(l, x);
        if !any(a.light > vec3<f32>(0.0)) {
            continue;
        }
        let origin = x + a.wi * ray_offset(t);
        var visible = vec3<f32>(gobo(l, x, u32(l.normal.w)) * medium_transmittance(origin, a.wi, a.reach));
        if shadows {
            visible *= transmittance(origin, a.wi, a.reach);
        }
        // Light travels along −wi and leaves toward the camera along −d.
        let cos_theta = dot(a.wi, d);
        var phased = vec3<f32>(0.0);
        bits = mask;
        while bits != 0u {
            let v = firstTrailingBit(bits);
            bits &= bits - 1u;
            phased += density[v] * media[v].scatter.rgb * phase(media[v].scale.w, cos_theta);
        }
        phased += fog_sigma * frame.fog_color.rgb * phase(frame.volume.w, cos_theta);
        lit += a.light * phased * visible;
    }
    // Ambient light from all round: the sky's and the ground's mean, where
    // this step's sky ray gets out.
    let count = max(u32(frame.volume.x), 1u);
    var sky = 1.0;
    if shadows {
        let up = frame.world_up.xyz;
        let dir = tangent_frame(up) * hemisphere_direction((*m).steps % count, count);
        sky = select(1.0, 0.0, occluded(x + up * ray_offset(t), dir, SKY));
    }
    let ambient = 0.5 * (frame.ambient.rgb + frame.ground.rgb) * sky;
    let source = lit + scatter * ambient + emitted;
    let kept = exp(-sigma * delta);
    (*m).light += (*m).transmittance * source / sigma * (1.0 - kept);
    (*m).transmittance *= kept;
    (*m).steps += 1u;
}

// Steps `n` over [a, b] at the pixel's jitter `j`.
fn stretch(m: ptr<function, March>, o: vec3<f32>, d: vec3<f32>, a: f32, b: f32, n: u32, j: f32, mask: u32, fog: bool) {
    let delta = (b - a) / f32(n);
    for (var k = 0u; k < n; k++) {
        if (*m).transmittance < 1e-3 {
            return;
        }
        let t = a + (f32(k) + j) * delta;
        march_step(m, o + d * t, t, delta, d, mask, fog);
    }
}

// Steps for a stretch `length` long of `total` such: its share of
// volume_steps, at least one.
fn steps_for(length: f32, total: f32) -> u32 {
    let n = f32(frame.volume.x);
    return u32(clamp(ceil(n * length / max(total, 1e-9)), 1.0, n));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.xy;
    if !inside_film(p) {
        return;
    }
    let fog = frame.media.z == 1u;
    var m = March(vec3<f32>(0.0), 1.0, 0u);
    if frame.media.x > 0u || fog {
        let o = ray_origin(p);
        let ray = ray_direction(p);
        let scale = length(ray);
        let d = ray / scale;
        var limit = frame.volume.y;
        if textureLoad(id_tex, p, 0).x != MISS {
            limit = min(limit, textureLoad(depth_tex, p, 0).x * scale);
        }
        let c = crossings(o, d);
        let fog_from = select(limit, min(frame.fog.x, limit), fog);
        // Two walks over the stretches: their lengths, then the march.
        var inside = 0.0;
        var outside = 0.0;
        let j = jitter(p);
        for (var walk = 0u; walk < 2u; walk++) {
            var mask = c.inside;
            var a = 0.0;
            for (var i = 0u; i <= c.count; i++) {
                var b = limit;
                if i < c.count {
                    b = min(c.t[i], limit);
                }
                if mask != 0u && b > a {
                    if walk == 0u {
                        inside += b - a;
                    } else {
                        stretch(&m, o, d, a, b, steps_for(b - a, inside), j, mask, fog);
                    }
                } else if fog && b > max(a, fog_from) {
                    let start = max(a, fog_from);
                    if walk == 0u {
                        outside += b - start;
                    } else {
                        stretch(&m, o, d, start, b, steps_for(b - start, outside), j, 0u, true);
                    }
                }
                if i == c.count || c.t[i] >= limit {
                    break;
                }
                mask ^= 1u << c.volume[i];
                a = c.t[i];
            }
        }
    }
    textureStore(volume_out, p, vec4<f32>(m.light, 1.0 - m.transmittance));
}
