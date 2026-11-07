// The bounce pass: one bounce of diffuse light (global illumination). Each
// film pixel traces gi samples rays over its cosine hemisphere (the fixed
// pattern of hemisphere.wgsl) up to gi distance; where one hits, the light
// there (light.wgsl: the lights, their shadows and gobos, through the hit's
// back where it is translucent) × the hit's albedo is what that ray brings
// back, and an unlit hit's albedo is light it gives. Their mean is radiance off a white Lambert surface, × gi. GI 0
// traces nothing and writes 0.

@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var id_tex: texture_2d<u32>;
@group(0) @binding(4) var indirect_out: texture_storage_2d<rgba16float, write>;

// The light a ray along `d` from `origin` brings back: what lights the
// closest surface within gi distance, × its albedo (Lambert, so / π comes
// at the end); 0 when it hits nothing. `depth` is the origin's, for the
// hit's ray offset.
fn bounce(origin: vec3<f32>, d: vec3<f32>, depth: f32) -> vec3<f32> {
    let hit = closest(origin, d, frame.gi.y);
    if hit.t >= frame.gi.y {
        return vec3<f32>(0.0);
    }
    let tri = triangles[hit.triangle];
    let object = instances[hit.instance];
    // The triangle's mean tint rides in its w's.
    let tint = vec3<f32>(tri.v0.w, tri.e1.w, tri.e2.w);
    if object.transmit.w > 0.5 {
        return PI * object.albedo.rgb * tint;
    }
    // The face the ray arrives at.
    var n = normalize(hit_normal(hit));
    n = select(n, -n, dot(n, d) > 0.0);
    let x = origin + d * hit.t;
    let light = irradiance(x, n, ray_offset(depth + hit.t), object.transmit.rgb);
    return light * object.albedo.rgb * tint;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.xy;
    if !inside_film(p) {
        return;
    }
    var light = vec3<f32>(0.0);
    if textureLoad(id_tex, p, 0).x != MISS && frame.gi.x > 0.0 {
        let depth = textureLoad(depth_tex, p, 0).x;
        let n = textureLoad(normal_tex, p, 0).xyz;
        let origin = world_position(p, depth) + n * ray_offset(depth);
        let basis = tangent_frame(n);
        let count = u32(frame.gi.z);
        for (var k = 0u; k < count; k++) {
            light += bounce(origin, basis * hemisphere_direction(k, count), depth);
        }
        light *= frame.gi.x / (PI * f32(count));
    }
    textureStore(indirect_out, p, vec4<f32>(light, 1.0));
}
