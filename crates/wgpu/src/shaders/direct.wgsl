// The direct-light pass: each film pixel's irradiance from the lights
// (light.wgsl), through its back face where the material is translucent,
// as radiance off a white Lambert surface, darkened by AO as much as AO
// on Direct asks.

@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var id_tex: texture_2d<u32>;
// Binding 4: the lights (light.wgsl).
@group(0) @binding(5) var transmit_tex: texture_2d<f32>;
@group(0) @binding(6) var ao_tex: texture_2d<f32>;
@group(0) @binding(7) var direct_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.xy;
    if !inside_film(p) {
        return;
    }
    var light = vec3<f32>(0.0);
    if textureLoad(id_tex, p, 0).x != MISS {
        let depth = textureLoad(depth_tex, p, 0).x;
        let n = textureLoad(normal_tex, p, 0).xyz;
        let transmit = textureLoad(transmit_tex, p, 0).rgb;
        let surface = world_position(p, depth);
        let contact = mix(1.0, textureLoad(ao_tex, p, 0).x, frame.ao.z);
        light = irradiance(surface, n, ray_offset(depth), transmit) * (contact / PI);
    }
    textureStore(direct_out, p, vec4<f32>(light, 1.0));
}
