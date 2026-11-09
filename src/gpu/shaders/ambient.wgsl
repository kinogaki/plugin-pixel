// The ambient pass: ambient occlusion from AO_RAYS hemisphere rays (the
// fixed pattern of hemisphere.wgsl) against the BVH up to ao_distance,
// shaped by AO Power and Strength (mix(1, open^power, strength)), times the domes'
// light: their radiance from above and from below the horizon (the same
// unless a dome has a ground color) blended by how far the normal faces
// up, n·up × ½ + ½ (for a two-color dome, the exact irradiance's share
// from above). Domes a portal names light through their portals alone:
// theirs counts the share of the open rays that leave through one (the
// portals follow the lights in the light buffer, frame.ao.w of them). No
// dome: no ambient light. Runs before the direct pass, which reads the AO.

@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var id_tex: texture_2d<u32>;
@group(0) @binding(4) var ambient_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var ao_out: texture_storage_2d<r32float, write>;

// Whether a ray from `o` along `d` leaves through a portal, against the way
// its light passes (`o` where it passes into).
fn through_portal(o: vec3<f32>, d: vec3<f32>) -> bool {
    var through = false;
    let first = frame.film.w;
    for (var i = 0u; i < u32(frame.ao.w); i++) {
        let p = lights[first + i];
        let facing = dot(d, p.normal.xyz);
        let t = dot(p.position.xyz - o, p.normal.xyz) / facing;
        let q = o + d * t - p.position.xyz;
        let u = p.axis.xyz;
        let v = p.power.xyz;
        let inside = abs(dot(q, u) / dot(u, u)) <= 1.0 && abs(dot(q, v) / dot(v, v)) <= 1.0;
        through = through || (facing < 0.0 && t > 0.0 && inside);
    }
    return through;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.xy;
    if !inside_film(p) {
        return;
    }
    var ao = 1.0;
    var share = 0.0;
    let n = textureLoad(normal_tex, p, 0).xyz;
    let reach = frame.ambient.w;
    let portals = frame.ao.w > 0.0;
    if textureLoad(id_tex, p, 0).x == MISS {
        ao = 0.0;
    } else if reach > 0.0 || portals {
        let depth = textureLoad(depth_tex, p, 0).x;
        let origin = world_position(p, depth) + n * ray_offset(depth);
        let basis = tangent_frame(n);
        var open = 0u;
        var through = 0u;
        for (var k = 0u; k < AO_RAYS; k++) {
            // A select, not a branch on the ray (see light.wgsl).
            let d = basis * hemisphere_direction(k, AO_RAYS);
            let free = reach <= 0.0 || !occluded(origin, d, reach);
            open += select(0u, 1u, free);
            through += select(0u, 1u, free && portals && through_portal(origin, d));
        }
        if reach > 0.0 {
            let traced = f32(open) / f32(AO_RAYS);
            ao = mix(1.0, pow(traced, frame.ao.y), frame.ao.x);
        }
        share = f32(through) / f32(max(open, 1u));
    }
    textureStore(ao_out, p, vec4<f32>(ao, 0.0, 0.0, 0.0));
    let up = dot(n, frame.world_up.xyz) * 0.5 + 0.5;
    let light = mix(frame.ground.rgb, frame.ambient.rgb, up)
        + mix(frame.through_ground.rgb, frame.through.rgb, up) * share;
    textureStore(ambient_out, p, vec4<f32>(light * ao, 1.0));
}
