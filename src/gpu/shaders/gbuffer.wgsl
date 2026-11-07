// The G-buffer pass: rasterizes the instances (each mesh drawn once for all
// its instances, in its own space, placed by each) at film resolution into albedo
// (the material's color × the vertex tint, its color attribute: blended
// across a triangle, flat where the tint is a primitive's, or its instance's
// own color, × its texture:
// the picture at the transformed coordinates, its nearest texel or four
// blended, each axis wrapped, × its factor; where the texture's alpha × the
// vertex alpha is below a cutout, nothing), normal (world, facing the camera), depth (along
// the view direction), object id and the material's transmittance (light
// through its back face; w: 1 when unlit).

@group(0) @binding(1) var atlas: texture_2d<f32>;

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

// Texel `i` of `n` along an axis wrapped by `mode` (0 repeat, 1 clamp, 2
// mirror, 3 black: -1 outside).
fn wrapped(i: i32, n: i32, mode: u32) -> i32 {
    switch mode {
        case 1u: {
            return clamp(i, 0, n - 1);
        }
        case 2u: {
            let m = ((i % (2 * n)) + 2 * n) % (2 * n);
            return select(m, 2 * n - 1 - m, m >= n);
        }
        case 3u: {
            return select(i, -1, i < 0 || i >= n);
        }
        default: {
            return ((i % n) + n) % n;
        }
    }
}

// Texel `(x, y)` of `object`'s picture, wrapped (rows from the top).
fn texel(object: Instance, size: vec2<i32>, x: i32, y: i32) -> vec4<f32> {
    let flags = object.texture.w;
    let wx = wrapped(x, size.x, (flags >> 2u) & 3u);
    let wy = wrapped(y, size.y, (flags >> 4u) & 3u);
    if wx < 0 || wy < 0 {
        return vec4<f32>(0.0);
    }
    let corner = vec2<i32>(object.texture.xy);
    return textureLoad(atlas, corner + vec2<i32>(wx, wy), 0);
}

// `object`'s texture at corner coordinates `uv` (MaterialX's, v up), × its
// factor; white without one.
fn textured(object: Instance, uv: vec2<f32>) -> vec4<f32> {
    let flags = object.texture.w;
    if (flags & 1u) == 0u {
        return vec4<f32>(1.0);
    }
    let size = vec2<i32>(i32(object.texture.z & 0xffffu), i32(object.texture.z >> 16u));
    let st = vec2<f32>(dot(object.uv_s.xyz, vec3<f32>(uv, 1.0)), dot(object.uv_t.xyz, vec3<f32>(uv, 1.0)));
    // The picture's rows run down from its top.
    let p = vec2<f32>(st.x, 1.0 - st.y) * vec2<f32>(size);
    var color: vec4<f32>;
    if (flags & 2u) != 0u {
        let at = vec2<i32>(floor(p));
        color = texel(object, size, at.x, at.y);
    } else {
        let q = p - 0.5;
        let f = fract(q);
        let at = vec2<i32>(floor(q));
        let top = mix(texel(object, size, at.x, at.y), texel(object, size, at.x + 1, at.y), f.x);
        let bottom = mix(texel(object, size, at.x, at.y + 1), texel(object, size, at.x + 1, at.y + 1), f.x);
        color = mix(top, bottom, f.y);
    }
    return color * object.factor;
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
    let picture = textured(object, v.uv);
    let tint = tinted(object, v.tint);
    if (object.texture.w & 64u) != 0u && picture.a * tint.a < object.uv_s.w {
        discard;
    }
    var out: Targets;
    out.albedo = vec4<f32>(object.albedo.rgb * tint.rgb * picture.rgb, 1.0);
    out.normal = vec4<f32>(n, 0.0);
    out.depth = dot(v.world - frame.eye.xyz, -frame.back.xyz);
    out.id = object.id.x;
    out.transmit = object.transmit;
    return out;
}
