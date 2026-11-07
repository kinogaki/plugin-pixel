// A surface where the G-buffer draws it or a ray meets it: its texture
// there (the picture in the atlas at the point's coordinates through the
// texture's placement, each axis wrapped, its nearest texel or four
// blended, × its factor) × its color and tint, and whether its cutout cuts
// it away there. The G-buffer and the traced rays share it (bvh.wgsl), so
// a ray passes where the camera sees the surface cut away. The atlas is
// the scene group's binding 9.

@group(1) @binding(9) var atlas: texture_2d<f32>;

// Texture flags (`texture.w`, records.rs): it samples one; the nearest
// texel; a cutout. Bits 2-3 and 4-5 are the wraps.
const TEXTURED: u32 = 1u;
const NEAREST: u32 = 2u;
const CUTOUT: u32 = 64u;

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
    if (flags & TEXTURED) == 0u {
        return vec4<f32>(1.0);
    }
    let size = vec2<i32>(i32(object.texture.z & 0xffffu), i32(object.texture.z >> 16u));
    let st = vec2<f32>(dot(object.uv_s.xyz, vec3<f32>(uv, 1.0)), dot(object.uv_t.xyz, vec3<f32>(uv, 1.0)));
    // The picture's rows run down from its top.
    let p = vec2<f32>(st.x, 1.0 - st.y) * vec2<f32>(size);
    var color: vec4<f32>;
    if (flags & NEAREST) != 0u {
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

// `object` where its corners blend to texture coordinates `uv` and tint
// `corners` (rgb, alpha): rgb its albedo there, its color × the tint × the
// texture; a its alpha there, the texture's × the tint's.
fn surface_at(object: Instance, uv: vec2<f32>, corners: vec4<f32>) -> vec4<f32> {
    let picture = textured(object, uv);
    let tint = tinted(object, corners);
    return vec4<f32>(object.albedo.rgb * tint.rgb * picture.rgb, picture.a * tint.a);
}

// Whether `object`'s cutout cuts it away where its alpha is `alpha`.
fn cut_away(object: Instance, alpha: f32) -> bool {
    return (object.texture.w & CUTOUT) != 0u && alpha < object.uv_s.w;
}
