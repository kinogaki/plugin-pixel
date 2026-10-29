// The ambient pass: ambient occlusion from AO_RAYS hemisphere rays (the
// fixed pattern of hemisphere.wgsl) against the BVH up to ao_distance,
// shaped by AO Power and Strength (shading::occlusion), times the domes'
// light: their radiance from above and from below the horizon (the same
// unless a dome has a ground color) blended by how far the normal faces
// up, n·up × ½ + ½ (for a two-color dome, the exact irradiance's share
// from above). No dome: no ambient light. Runs before the direct pass,
// which reads the AO.

@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var id_tex: texture_2d<u32>;
@group(0) @binding(4) var ambient_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var ao_out: texture_storage_2d<r32float, write>;

// AO_RAYS comes from lighting::AO_RAYS.

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.xy;
    if !inside_film(p) {
        return;
    }
    var ao = 1.0;
    let n = textureLoad(normal_tex, p, 0).xyz;
    if textureLoad(id_tex, p, 0).x == MISS {
        ao = 0.0;
    } else if frame.ambient.w > 0.0 {
        let depth = textureLoad(depth_tex, p, 0).x;
        let origin = world_position(p, depth) + n * ray_offset(depth);
        let basis = tangent_frame(n);
        var open = 0u;
        for (var k = 0u; k < AO_RAYS; k++) {
            // A select, not a branch on the ray (see light.wgsl).
            let d = basis * hemisphere_direction(k, AO_RAYS);
            open += select(1u, 0u, occluded(origin, d, frame.ambient.w));
        }
        let traced = f32(open) / f32(AO_RAYS);
        ao = mix(1.0, pow(traced, frame.ao.y), frame.ao.x);
    }
    textureStore(ao_out, p, vec4<f32>(ao, 0.0, 0.0, 0.0));
    let up = dot(n, frame.world_up.xyz) * 0.5 + 0.5;
    let light = mix(frame.ground.rgb, frame.ambient.rgb, up);
    textureStore(ambient_out, p, vec4<f32>(light * ao, 1.0));
}
