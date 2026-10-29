// The per-frame uniform every pass reads (FrameUniform in gpu/frame.rs).

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
    // shade_levels, dither_strength, Shade Ramp stops; unused.
    shade: vec4<f32>,
    // rgb: the rim lights shown where direct light reaches, summed.
    rim: vec4<f32>,
    // rgb: the rim lights shown all round, summed.
    rim_all: vec4<f32>,
    // rgb: the fog's color; w: its density (0: no fog).
    fog_color: vec4<f32>,
    // The fog's start, height, falloff, unused.
    fog: vec4<f32>,
    // AO's strength, power and share on direct light; unused.
    ao: vec4<f32>,
    // GI's strength, distance and samples; unused.
    gi: vec4<f32>,
    // Volume steps, max distance, 1 with shadow rays; lit fog's anisotropy.
    volume: vec4<f32>,
    // Volumes, how many cast shadows, 1 when the fog is lit; unused.
    media: vec4<u32>,
    // The volumes' and fog's Volume Levels and Volume Dither; unused.
    bands: vec4<f32>,
    // The dither's thresholds, row-major, four a vector (8×8 at most).
    bayer: array<vec4<f32>, 16>,
    // The Shade Ramp's stops: rgb, and w the position (shade.w of them).
    ramp: array<vec4<f32>, 16>,
}

// An object table entry (geometry::Object): its material and id.
struct Object {
    // rgb: the Pixel Surface's color (ACEScg).
    albedo: vec4<f32>,
    // rgb: its transmittance, translucency × translucency color.
    transmit: vec4<f32>,
    // x: the object id.
    id: vec4<u32>,
}

// The id of a film pixel no mesh covers.
const MISS: u32 = 0xffffffffu;
const PI: f32 = 3.14159265358979;

@group(0) @binding(0) var<uniform> frame: Frame;

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
