// Direct light at a surface point from every light (gpu/lights.rs packs
// them), one hard shadow ray each, shared by the direct pass and the
// bounce pass's secondary hits. Each is a point at its centre of its
// shape's power (a distant light parallel light):
// - area lights: falloff 1 / (d² + r²), times their cosine toward the point;
// - spheres: falloff 1 / max(d², r²) (a point's 1 / d²);
// - cylinders: falloff 1 / max(d², r²), times the sine to their axis.
// ShapingAPI shapes each but a distant light by the direction it leaves in
// against its -Z: the cone's smoothstep over angles, focus toward its tint,
// an IES profile read at PROFILE vertical angles. A gobo masks distant and
// sphere lights, on the light's plane. Light on the back of a thin surface
// comes through × its transmittance, its shadow ray leaving from the back;
// a shadow ray keeps what the thin surfaces it crosses let through. A light
// whose shadows are ShadowAPI's: occluders past its shadow distance cast
// none, those within its falloff of it fade out, and what they shadow takes
// its shadow color. Every shadow ray first steps toward its light. The
// volume pass reads each light's arrival the same way. Comes after
// medium.wgsl.
//
// The lights are the scene group's binding 7.

const PROFILE: u32 = 16u;

struct Light {
    // xyz: the centre; w: the falloff's r².
    position: vec4<f32>,
    // xyz: where it shines (the light's -Z), unit; w: the kind.
    normal: vec4<f32>,
    // rgb: power (area), illuminance (distant) or intensity (sphere, cylinder).
    power: vec4<f32>,
    // xyz: a cylinder's axis (its X), unit.
    axis: vec4<f32>,
    // x: the cone's angle (radians, π none); y: where its softness starts;
    // z: focus (0 none); w: the profile's angle scale.
    shaping: vec4<f32>,
    // rgb: the focus tint; w: 1 with a profile.
    tint: vec4<f32>,
    // The profile's candela at (k + ½) / PROFILE × π off the axis, four a row.
    profile: array<vec4<f32>, 4>,
    // rgb: the shadow color; w: the shadow distance (-1 any, 0 none).
    shadow: vec4<f32>,
    // x: the shadow falloff (-1 none); y: its gamma; z: the gobo's contrast;
    // w: 1 with a gobo.
    fade: vec4<f32>,
    // The gobo's plane: xyz the light's X (Y) over the scale; w the offset over it.
    gobo_x: vec4<f32>,
    gobo_y: vec4<f32>,
    // x: the gobo's seed.
    seed: vec4<u32>,
}

const AREA: u32 = 0u;
const DISTANT: u32 = 1u;
const SPHERE: u32 = 2u;
const CYLINDER: u32 = 3u;
const LIGHT_PI: f32 = 3.14159265;
// Ray offsets a shadow ray steps toward its light before it starts.
const SHADOW_STEP: f32 = 16.0;
// Far enough for a distant light's shadow ray to leave the scene.
const FAR_LIGHT: f32 = 1e30;

@group(1) @binding(7) var<storage, read> lights: array<Light>;

// ShapingAPI's cone at `theta` off the axis: whole up to its softness's
// start, a smoothstep to nothing at its angle.
fn cone_factor(theta: f32, start: f32, cone: f32) -> f32 {
    if cone - start <= 1e-7 {
        return select(1.0, 0.0, theta > cone);
    }
    let t = clamp((theta - start) / (cone - start), 0.0, 1.0);
    return 1.0 - t * t * (3.0 - 2.0 * t);
}

// The profile's candela at `theta` (radians off the axis) after its angle
// scale, linear between its angles, held at its ends.
fn profile_at(l: Light, theta: f32) -> f32 {
    let scale = l.shaping.w;
    var t = theta;
    if scale > 0.0 {
        t = theta / scale;
    } else if scale < 0.0 {
        t = (theta - LIGHT_PI) / -scale + LIGHT_PI;
    }
    if t < 0.0 {
        return 0.0;
    }
    let at = clamp(min(t, LIGHT_PI) / LIGHT_PI * f32(PROFILE) - 0.5, 0.0, f32(PROFILE - 1u));
    let k = u32(floor(at));
    let next = min(k + 1u, PROFILE - 1u);
    let a = l.profile[k / 4u][k % 4u];
    let b = l.profile[next / 4u][next % 4u];
    return mix(a, b, at - f32(k));
}

// What light `l` sends toward unit direction `e` (leaving it) is
// multiplied by: its cone, focus and profile.
fn shaped(l: Light, e: vec3<f32>) -> vec3<f32> {
    let c = clamp(dot(e, l.normal.xyz), -1.0, 1.0);
    let theta = acos(c);
    var f = vec3<f32>(cone_factor(theta, l.shaping.y, l.shaping.x));
    if l.shaping.z > 0.0 {
        f *= mix(l.tint.rgb, vec3<f32>(1.0), pow(abs(c), l.shaping.z));
    }
    if l.tint.w > 0.0 {
        f *= profile_at(l, theta);
    }
    return f;
}

// The gobo at world point `x`: projected along the light onto its plane
// (a sphere's at unit distance ahead of it).
fn gobo(l: Light, x: vec3<f32>, kind: u32) -> f32 {
    if l.fade.w == 0.0 {
        return 1.0;
    }
    let d = x - l.position.xyz;
    var across = vec2<f32>(dot(d, l.gobo_x.xyz), dot(d, l.gobo_y.xyz));
    if kind == SPHERE {
        across /= max(dot(d, l.normal.xyz), 1e-6);
    }
    return gobo_mask(across + vec2<f32>(l.gobo_x.w, l.gobo_y.w), l.fade.z, l.seed.x);
}

// What light `l` brings to world point `x` before any cosine, shadow or
// gobo: the unit direction toward it, how far a shadow ray looks, and the
// light (a distant light's illuminance; a point's, fallen off and shaped).
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
        if kind == AREA {
            a.light *= max(-dot(l.normal.xyz, a.wi), 0.0) / (d2 + l.position.w);
        } else {
            a.light /= max(d2, max(l.position.w, 1e-12));
            if kind == CYLINDER {
                a.light *= length(cross(a.wi, l.axis.xyz));
            }
        }
        a.light *= shaped(l, -a.wi);
        a.reach = reach * (1.0 - 1e-4);
    }
    return a;
}

// Whether light `l`'s shadows are ShadowAPI's (other than black at any
// distance).
fn shadow_shaped(l: Light) -> bool {
    return l.shadow.w >= 0.0 || any(l.shadow.rgb > vec3<f32>(0.0));
}

// What of light `l` reaches past ShadowAPI's shadows along unit `wi` from
// `origin`, `reach` to the light: occluders past its shadow distance cast
// none, within its falloff of it they fade out by its gamma, and what they
// shadow takes its shadow color (thin surfaces shadow these as opaque).
fn shadowed(l: Light, origin: vec3<f32>, wi: vec3<f32>, reach: f32) -> vec3<f32> {
    let limit = l.shadow.w;
    if limit == 0.0 {
        return vec3<f32>(1.0);
    }
    var far = reach;
    if limit > 0.0 {
        far = min(reach, limit);
    }
    let hit = closest(origin, wi, far, l.seed.y);
    if hit.t >= far {
        return vec3<f32>(1.0);
    }
    var strength = 1.0;
    let falloff = l.fade.x;
    if limit > 0.0 && falloff > 0.0 {
        let start = limit - min(falloff, limit);
        strength = pow(clamp((limit - hit.t) / max(limit - start, 1e-20), 0.0, 1.0), l.fade.y);
    }
    return mix(vec3<f32>(1.0), l.shadow.rgb, strength);
}

// What of light `l` reaches `origin` along unit `wi`, `reach` to it, past
// what shadows it: the thin surfaces' transmittance, or ShadowAPI's.
fn unshadowed(l: Light, origin: vec3<f32>, wi: vec3<f32>, reach: f32) -> vec3<f32> {
    if shadow_shaped(l) {
        return shadowed(l, origin, wi, reach);
    }
    return transmittance(origin, wi, reach, l.seed.y);
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
            let visible = unshadowed(l, origin, a.wi, a.reach)
                * medium_transmittance(origin, a.wi, a.reach);
            let through = select(vec3<f32>(1.0), transmit, back);
            light += a.light * through * visible * (abs(cos_s) * gobo(l, surface, u32(l.normal.w)));
        }
    }
    return light;
}
