// The G-buffer pass: rasterizes the instances (each mesh drawn once for all
// its instances, in its own space, placed by each) at film resolution into
// albedo (the surface at the fragment, surface.wgsl: the material's color
// × the vertex tint, its color attribute: blended across a triangle, flat
// where the tint is a primitive's, or its instance's own color, × its
// texture; where its cutout cuts it away, nothing), normal (world, facing
// the camera), depth (along the view direction), object id and the
// material's transmittance (light through its back face; w: 1 when unlit).

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // rgb: what the albedo is multiplied by; a: the texture's alpha.
    @location(2) tint: vec4<f32>,
    // The instance placing it (one a drawn instance).
    @location(3) instance: u32,
    @location(4) uv: vec2<f32>,
}

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) instance: u32,
    @location(3) tint: vec4<f32>,
    @location(4) uv: vec2<f32>,
}

struct Targets {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) depth: f32,
    @location(3) id: u32,
    @location(4) transmit: vec4<f32>,
}

@vertex
fn vs(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    let i = instances[v.instance];
    let world = own_to_world(i, v.position);
    out.clip = frame.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.normal = own_to_world_normal(i, v.normal);
    out.instance = v.instance;
    out.tint = v.tint;
    out.uv = v.uv;
    return out;
}

@fragment
fn fs(v: VertexOut) -> Targets {
    let object = instances[v.instance];
    // Toward the camera: the eye, or the view direction's reverse.
    let to_eye = select(frame.eye.xyz - v.world, frame.back.xyz, orthographic());
    var n = normalize(v.normal);
    // Two-sided: the normal of the side the camera sees.
    if dot(n, to_eye) < 0.0 {
        n = -n;
    }
    let seen = surface_at(object, v.uv, v.tint);
    if cut_away(object, seen.a) {
        discard;
    }
    var out: Targets;
    out.albedo = vec4<f32>(seen.rgb, 1.0);
    out.normal = vec4<f32>(n, 0.0);
    out.depth = dot(v.world - frame.eye.xyz, -frame.back.xyz);
    out.id = object.id.x;
    out.transmit = object.transmit;
    return out;
}
