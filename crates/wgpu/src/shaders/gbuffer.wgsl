// The G-buffer pass: rasterizes the meshes at film resolution into albedo
// (the material's color × the vertex tint, its color attribute: blended
// across a triangle, flat where the tint is a primitive's), normal (world,
// facing the camera), depth (along the view direction), object id and the
// material's transmittance (light through its back face).

@group(0) @binding(1) var<storage, read> objects: array<Object>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) object: u32,
    @location(2) normal: vec3<f32>,
    @location(3) tint: vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) object: u32,
    @location(3) tint: vec3<f32>,
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
    out.clip = frame.view_proj * vec4<f32>(v.position, 1.0);
    out.world = v.position;
    out.normal = v.normal;
    out.object = v.object;
    out.tint = v.tint;
    return out;
}

@fragment
fn fs(v: VertexOut) -> Targets {
    let object = objects[v.object];
    // Toward the camera: the eye, or the view direction's reverse.
    let to_eye = select(frame.eye.xyz - v.world, frame.back.xyz, orthographic());
    var n = normalize(v.normal);
    // Two-sided: the normal of the side the camera sees.
    if dot(n, to_eye) < 0.0 {
        n = -n;
    }
    var out: Targets;
    out.albedo = vec4<f32>(object.albedo.rgb * v.tint, 1.0);
    out.normal = vec4<f32>(n, 0.0);
    out.depth = dot(v.world - frame.eye.xyz, -frame.back.xyz);
    out.id = object.id.x;
    out.transmit = vec4<f32>(object.transmit.rgb, 0.0);
    return out;
}
