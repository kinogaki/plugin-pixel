// The per-frame uniform every pass reads (Frame in gpu/records.rs), and the
// object table the G-buffer and the rays read (the scene's group 1).

struct Frame {
    view_proj: mat4x4<f32>,
    // xyz: the eye (orthographic: the centre of the camera's plane); w:
    // the ray origin offset at zero depth.
    eye: vec4<f32>,
    // Camera axes; w: the film's top-left corner in film space (x, y) and
    // the film-space extent of one film pixel (camera.rs: tangents, or
    // lengths across the plane when orthographic).
    right: vec4<f32>,
    up: vec4<f32>,
    back: vec4<f32>,
    // rgb: the domes' radiance from above the horizon; w: ao_distance.
    ambient: vec4<f32>,
    // rgb: the domes' radiance from below it (the ground colors'); w: 1
    // when the camera sees the domes behind the scene.
    ground: vec4<f32>,
    // xyz: the world's up, unit.
    world_up: vec4<f32>,
    // Film width, height, pixel size, light count.
    film: vec4<u32>,
    // View width, height, dither size (0: off), 1 when orthographic.
    view: vec4<u32>,
    // Shade Levels, Dither Strength, Volume Levels, Volume Dither.
    shade: vec4<f32>,
    // rgb: the rim lights shown where direct light reaches, summed.
    rim: vec4<f32>,
    // rgb: the rim lights shown all round, summed.
    rim_all: vec4<f32>,
    // rgb: the fog's color; w: its density (0: no fog).
    fog_color: vec4<f32>,
    // The fog's start, height, falloff and (lit) anisotropy.
    fog: vec4<f32>,
    // AO's strength, power and share on direct light; unused.
    ao: vec4<f32>,
    // GI's strength, distance and samples; unused.
    gi: vec4<f32>,
    // Volume steps, max distance, 1 with shadow rays; unused.
    volume: vec4<f32>,
    // Volumes, how many cast shadows, 1 when the fog is lit; unused.
    media: vec4<u32>,
    // The dither's thresholds, row-major, four a vector (8×8 at most).
    bayer: array<vec4<f32>, 16>,
    // The Shade Ramp at each shade level, darkest first (rgb).
    shade_ramp: array<vec4<f32>, 64>,
    // The Shade Ramp at each volume level.
    volume_ramp: array<vec4<f32>, 64>,
}

// A mesh placed (records::GpuInstance): its own space to world and back,
// three rows each (a flat placement's way back is NaN: no ray enters it),
// its material and ids.
struct Instance {
    to_world_x: vec4<f32>,
    to_world_y: vec4<f32>,
    to_world_z: vec4<f32>,
    to_own_x: vec4<f32>,
    to_own_y: vec4<f32>,
    to_own_z: vec4<f32>,
    // rgb: the Pixel Surface's color (ACEScg).
    albedo: vec4<f32>,
    // rgb: its transmittance, translucency × translucency color; w: 1 when
    // it is unlit (its color whatever lights it).
    transmit: vec4<f32>,
    // x: the object id; y: its mesh's BVH root node; z: what of `tint`
    // stands for its corners' (1 its color, 2 its alpha).
    id: vec4<u32>,
    // Its texture's picture in the atlas: x, y its corner; z its width and
    // height (16 bits each); w the flags (records.rs: 1 textured, 2 the
    // nearest texel, bits 2-3 and 4-5 the wraps along s and t: repeat,
    // clamp, mirror, black; 64 a cutout).
    texture: vec4<u32>,
    // The texture's transform, by rows; uv_s.w: the cutout's alpha.
    uv_s: vec4<f32>,
    uv_t: vec4<f32>,
    // What a texel is multiplied by.
    factor: vec4<f32>,
    // rgb: the albedo a ray bouncing off it sees (its color × its texture's
    // mean).
    mean: vec4<f32>,
    // What stands for its mesh's corners' tint (rgb) and alpha (w): its
    // instance's own, where `id.z` says it holds them.
    tint: vec4<f32>,
}

// The tint and alpha of `object` where its corners' are `corners`: its own
// where it has them.
fn tinted(object: Instance, corners: vec4<f32>) -> vec4<f32> {
    let color = select(corners.rgb, object.tint.rgb, (object.id.z & 1u) != 0u);
    return vec4<f32>(color, select(corners.a, object.tint.a, (object.id.z & 2u) != 0u));
}

// The id of a film pixel no mesh covers (the standard objectId's miss).
const MISS: u32 = 0xffffffffu;
const PI: f32 = 3.14159265358979;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(6) var<storage, read> instances: array<Instance>;

// Own-space point `p` of instance `i` in world space.
fn own_to_world(i: Instance, p: vec3<f32>) -> vec3<f32> {
    let h = vec4<f32>(p, 1.0);
    return vec3<f32>(dot(i.to_world_x, h), dot(i.to_world_y, h), dot(i.to_world_z, h));
}

// Own-space normal `n` of instance `i` in world space (not unit): by the
// inverse transpose, the way back's rows as columns.
fn own_to_world_normal(i: Instance, n: vec3<f32>) -> vec3<f32> {
    return i.to_own_x.xyz * n.x + i.to_own_y.xyz * n.y + i.to_own_z.xyz * n.z;
}

fn orthographic() -> bool {
    return frame.view.w == 1u;
}

// Film pixel `p`'s centre in film space.
fn film_space(p: vec2<u32>) -> vec2<f32> {
    let f = vec2<f32>(p) + 0.5;
    return vec2<f32>(frame.right.w + f.x * frame.back.w, frame.up.w - f.y * frame.back.w);
}

// Where the primary ray through film pixel `p` starts: the eye, or its
// point on an orthographic camera's plane.
fn ray_origin(p: vec2<u32>) -> vec3<f32> {
    let t = film_space(p);
    let across = frame.right.xyz * t.x + frame.up.xyz * t.y;
    return frame.eye.xyz + select(vec3<f32>(0.0), across, orthographic());
}

// The primary ray's direction (not unit: one unit of depth a unit of it):
// parallel to the view direction when orthographic.
fn ray_direction(p: vec2<u32>) -> vec3<f32> {
    let t = film_space(p);
    let across = frame.right.xyz * t.x + frame.up.xyz * t.y;
    return select(across, vec3<f32>(0.0), orthographic()) - frame.back.xyz;
}

// The world point film pixel `p` sees at `depth` along the view direction.
fn world_position(p: vec2<u32>, depth: f32) -> vec3<f32> {
    return ray_origin(p) + ray_direction(p) * depth;
}

// The unit direction from where film pixel `p`'s primary ray starts toward
// what it sees.
fn view_direction(p: vec2<u32>) -> vec3<f32> {
    return normalize(ray_direction(p));
}

// How far secondary rays start off a surface at `depth`.
fn ray_offset(depth: f32) -> f32 {
    return frame.eye.w + depth * 1e-5;
}

fn inside_film(p: vec2<u32>) -> bool {
    return p.x < frame.film.x && p.y < frame.film.y;
}
