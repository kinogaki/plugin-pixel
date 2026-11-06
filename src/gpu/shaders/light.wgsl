// Direct light at a surface point from every light (gpu/records.rs packs
// them), one hard shadow ray each, shared by the direct pass and the
// bounce pass's secondary hits:
// - area lights: a point emitter at the centre, falloff 1 / (d² + r²);
// - distant lights: parallel light of their illuminance, no falloff;
// - spots: a point at the centre, falloff 1 / max(d², r²), times the cone.
// A gobo masks distant lights and spots, on the light's plane. Light on the
// back of a thin surface comes through × its transmittance, its shadow ray
// leaving from the back; a shadow ray keeps what the thin surfaces it
// crosses let through. Every shadow ray first steps toward its light.
// The volume pass reads each light's arrival the same way. Comes after
// medium.wgsl.
//
// The lights are the scene group's binding 7.

struct Light {
    // xyz: the centre; w: the falloff's r².
    position: vec4<f32>,
    // xyz: where it shines (the light's -Z), unit; w: the kind.
    normal: vec4<f32>,
    // rgb: power (area), illuminance (distant) or intensity (spot).
    power: vec4<f32>,
    // x, y: the cone's cosines (outer, inner); z: gobo contrast; w: 1 with a gobo.
    cone: vec4<f32>,
    // The gobo's plane: xyz the light's X (Y) over the scale; w the offset over it.
    gobo_x: vec4<f32>,
    gobo_y: vec4<f32>,
    // x: the gobo's seed.
    seed: vec4<u32>,
}

const AREA: u32 = 0u;
const DISTANT: u32 = 1u;
const SPOT: u32 = 2u;
// Ray offsets a shadow ray steps toward its light before it starts.
const SHADOW_STEP: f32 = 16.0;
// Far enough for a distant light's shadow ray to leave the scene.
const FAR_LIGHT: f32 = 1e30;

@group(1) @binding(7) var<storage, read> lights: array<Light>;

// The render interface's cone_falloff, from its two cosines.
fn cone_falloff(cos_axis: f32, outer: f32, inner: f32) -> f32 {
    if inner - outer <= 1e-7 {
        return select(0.0, 1.0, cos_axis >= outer);
    }
    let t = clamp((cos_axis - outer) / (inner - outer), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// The gobo at world point `x`: projected along the light onto its plane
// (a spot's at unit distance ahead of it).
fn gobo(l: Light, x: vec3<f32>, kind: u32) -> f32 {
    if l.cone.w == 0.0 {
        return 1.0;
    }
    let d = x - l.position.xyz;
    var across = vec2<f32>(dot(d, l.gobo_x.xyz), dot(d, l.gobo_y.xyz));
    if kind == SPOT {
        across /= max(dot(d, l.normal.xyz), 1e-6);
    }
    return gobo_mask(across + vec2<f32>(l.gobo_x.w, l.gobo_y.w), l.cone.z, l.seed.x);
}

// What light `l` brings to world point `x` before any cosine, shadow or
// gobo: the unit direction toward it, how far a shadow ray looks, and the
// light (a distant light's illuminance; a point's, fallen off and coned).
struct Arrival {
    wi: vec3<f32>,
    reach: f32,
    light: vec3<f32>,
}

fn arrival(l: Light, x: vec3<f32>) -> Arrival {
    let kind = u32(l.normal.w);
    var a = Arrival(-l.normal.xyz, FAR_LIGHT, l.power.rgb);
    if kind != DISTANT {
        let to_light = l.position.xyz - x;
        let d2 = dot(to_light, to_light);
        let reach = sqrt(d2);
        a.wi = to_light / reach;
        let cos_l = -dot(l.normal.xyz, a.wi);
        if kind == AREA {
            a.light *= max(cos_l, 0.0) / (d2 + l.position.w);
        } else {
            a.light *= cone_falloff(cos_l, l.cone.x, l.cone.y) / max(d2, l.position.w);
        }
        a.reach = reach * (1.0 - 1e-4);
    }
    return a;
}

// The irradiance at `surface` (unit normal `n`, the side seen), shadowed:
// Lambert light on the front, plus light on the back × `transmit`. Shadow
// rays start `offset` off the side the light is on, and keep what the
// shadowing volumes let through (medium.wgsl).
fn irradiance(surface: vec3<f32>, n: vec3<f32>, offset: f32, transmit: vec3<f32>) -> vec3<f32> {
    let thin = any(transmit > vec3<f32>(0.0));
    var light = vec3<f32>(0.0);
    for (var i = 0u; i < frame.film.w; i++) {
        let l = lights[i];
        let a = arrival(l, surface);
        let cos_s = dot(n, a.wi);
        let back = cos_s < 0.0 && thin;
        if (cos_s > 0.0 || back) && any(a.light > vec3<f32>(0.0)) {
            // Off the side the light is on, then a step toward the light:
            // a thin card shaded by an authored normal (Crown Normal's)
            // that leans past its own plane would otherwise shadow itself.
            let origin = surface + n * select(offset, -offset, back) + a.wi * (SHADOW_STEP * offset);
            // A multiplier, not a branch on the shadow ray: on Metal (M4
            // Max, wgpu 30) `if !occluded(..) { light += .. }` lit every
            // shadowed pixel, though occluded() returned true. Thin
            // translucent surfaces on the way let some through.
            let visible = transmittance(origin, a.wi, a.reach)
                * medium_transmittance(origin, a.wi, a.reach);
            let through = select(vec3<f32>(1.0), transmit, back);
            light += a.light * through * visible * (abs(cos_s) * gobo(l, surface, u32(l.normal.w)));
        }
    }
    return light;
}
