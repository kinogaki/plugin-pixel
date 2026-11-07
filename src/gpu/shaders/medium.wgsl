// Volumes (gpu/records.rs packs them): their bounds' BVH and records (a
// uniform of MAX_VOLUMES) at group 1, bindings 3 to 5, where a ray crosses
// them, their density, Henyey–Greenstein's phase, lit fog's
// density, and the transmittance a shadow ray keeps through the volumes
// that cast shadows. Comes after bvh.wgsl.

struct Medium {
    // World to the mesh's own space, three rows.
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    // xyz: the bounding box's minimum, own space; w: the edge falloff.
    lo: vec4<f32>,
    // xyz: its maximum; w: the density.
    hi: vec4<f32>,
    // xyz: scene units a unit of own space spans; w: the anisotropy.
    scale: vec4<f32>,
    // rgb: the scatter color; w: the noise amount.
    scatter: vec4<f32>,
    // rgb: emission × its color; w: the noise frequency.
    emission: vec4<f32>,
    // xyz: the noise offset.
    offset: vec4<f32>,
    // x: octaves; y: seed; z: 1 when it casts shadows.
    flags: vec4<u32>,
}

@group(1) @binding(3) var<storage, read> medium_nodes: array<Node>;
@group(1) @binding(4) var<storage, read> medium_triangles: array<Triangle>;
@group(1) @binding(5) var<uniform> media: array<Medium, MAX_VOLUMES>;

// The crossings a ray keeps (the nearest; parity counts them all).
const MAX_CROSSINGS: u32 = 16u;
// Steps a shadow ray takes through each stretch of a shadowing volume.
const SHADOW_STEPS: u32 = 4u;
// As far as rays look for crossings.
const FAR_MEDIUM: f32 = 1e30;

// Where a ray crosses volume bounds, nearest first, and which volumes its
// origin is inside (a bit a volume: a closed mesh crossed an odd number of
// times ahead).
struct Crossings {
    t: array<f32, MAX_CROSSINGS>,
    volume: array<u32, MAX_CROSSINGS>,
    count: u32,
    inside: u32,
}

fn crossings(o: vec3<f32>, d: vec3<f32>) -> Crossings {
    var c: Crossings;
    c.count = 0u;
    c.inside = 0u;
    if frame.media.x == 0u {
        return c;
    }
    let r = ray(o, d);
    var stack: array<u32, STACK>;
    var top = 1u;
    stack[0] = 0u;
    while top > 0u {
        top -= 1u;
        let n = medium_nodes[stack[top]];
        if box_entry(n, r, FAR_MEDIUM) < 0.0 {
            continue;
        }
        if n.count == 0u {
            stack[top] = n.a;
            stack[top + 1u] = n.a + 1u;
            top += 2u;
            continue;
        }
        for (var k = n.a; k < n.a + n.count; k++) {
            let tri = medium_triangles[k];
            let t = crossing(tri, r).x;
            if t <= 0.0 {
                continue;
            }
            let v = bitcast<u32>(tri.v0.w);
            // A ray through an edge meets both its triangles: one crossing.
            var twin = false;
            for (var i = 0u; i < c.count; i++) {
                twin = twin || (c.volume[i] == v && abs(c.t[i] - t) <= 1e-5 * max(t, 1.0));
            }
            if twin {
                continue;
            }
            c.inside ^= 1u << v;
            // Insert in order, dropping the farthest when full.
            if c.count == MAX_CROSSINGS && t >= c.t[MAX_CROSSINGS - 1u] {
                continue;
            }
            var at = min(c.count, MAX_CROSSINGS - 1u);
            while at > 0u && c.t[at - 1u] > t {
                c.t[at] = c.t[at - 1u];
                c.volume[at] = c.volume[at - 1u];
                at -= 1u;
            }
            c.t[at] = t;
            c.volume[at] = v;
            c.count = min(c.count + 1u, MAX_CROSSINGS);
        }
    }
    return c;
}

fn medium_hash(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    var h = (u32(x) * 0x8da6b343u) ^ (u32(y) * 0xd8163841u) ^ (u32(z) * 0xcb1ab31fu)
        ^ (seed * 0x9e3779b9u);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return f32(h >> 8u) / 16777216.0;
}

fn value_noise3(p: vec3<f32>, seed: u32) -> f32 {
    let f = floor(p);
    let i = vec3<i32>(f);
    let t = p - f;
    let s = t * t * (3.0 - 2.0 * t);
    let x00 = mix(medium_hash(i.x, i.y, i.z, seed), medium_hash(i.x + 1, i.y, i.z, seed), s.x);
    let x10 = mix(medium_hash(i.x, i.y + 1, i.z, seed), medium_hash(i.x + 1, i.y + 1, i.z, seed), s.x);
    let x01 = mix(medium_hash(i.x, i.y, i.z + 1, seed), medium_hash(i.x + 1, i.y, i.z + 1, seed), s.x);
    let x11 = mix(
        medium_hash(i.x, i.y + 1, i.z + 1, seed),
        medium_hash(i.x + 1, i.y + 1, i.z + 1, seed),
        s.x,
    );
    return mix(mix(x00, x10, s.y), mix(x01, x11, s.y), s.z);
}

fn fbm3(p: vec3<f32>, octaves: u32, seed: u32) -> f32 {
    var sum = 0.0;
    var total = 0.0;
    var weight = 0.5;
    var scale = 1.0;
    for (var octave = 0u; octave < clamp(octaves, 1u, 8u); octave++) {
        let shift = 17.0 * f32(octave);
        let q = vec3<f32>(p.x * scale + shift, p.y * scale - shift, p.z * scale + shift);
        sum += weight * value_noise3(q, seed + octave);
        total += weight;
        weight *= 0.5;
        scale *= 2.0;
    }
    return sum / total;
}

fn smooth01(t: f32) -> f32 {
    let u = clamp(t, 0.0, 1.0);
    return u * u * (3.0 - 2.0 * u);
}

// Medium `i`'s density at world point `x`, inside it.
fn medium_density(i: u32, x: vec3<f32>) -> f32 {
    let m = media[i];
    let p = vec4<f32>(x, 1.0);
    let local = vec3<f32>(dot(m.r0, p), dot(m.r1, p), dot(m.r2, p));
    let inside3 = min(local - m.lo.xyz, m.hi.xyz - local) * m.scale.xyz;
    let inside = min(min(inside3.x, inside3.y), inside3.z);
    var edge = 1.0;
    if m.lo.w > 0.0 {
        edge = smooth01(inside / m.lo.w);
    }
    var noise = 1.0;
    if m.scatter.w > 0.0 {
        let n = fbm3(x * m.emission.w + m.offset.xyz, m.flags.x, m.flags.y);
        noise = 1.0 + (smooth01((n - 0.3) / 0.4) - 1.0) * m.scatter.w;
    }
    return m.hi.w * edge * noise;
}

// Henyey–Greenstein at the cosine between the light's travel and the
// scattered direction.
fn phase(g: f32, cos_theta: f32) -> f32 {
    let g2 = g * g;
    let d = max(1.0 + g2 - 2.0 * g * cos_theta, 1e-6);
    return (1.0 - g2) / (4.0 * PI * d * sqrt(d));
}

// Lit fog's density at `x`, `t` along the camera ray (the shade pass's fog).
fn fog_density(x: vec3<f32>, t: f32) -> f32 {
    if t < frame.fog.x {
        return 0.0;
    }
    return frame.fog_color.w * exp(-frame.fog.z * (dot(x, frame.world_up.xyz) - frame.fog.y));
}

// What a ray from `o` along unit `d` keeps through the shadowing volumes
// up to `reach`: e^(−optical depth), SHADOW_STEPS midpoints a stretch.
fn medium_transmittance(o: vec3<f32>, d: vec3<f32>, reach: f32) -> f32 {
    if frame.media.y == 0u {
        return 1.0;
    }
    let c = crossings(o, d);
    var mask = c.inside;
    var a = 0.0;
    var depth = 0.0;
    for (var i = 0u; i <= c.count; i++) {
        var b = reach;
        if i < c.count {
            b = min(c.t[i], reach);
        }
        if b > a && mask != 0u {
            let step = (b - a) / f32(SHADOW_STEPS);
            for (var k = 0u; k < SHADOW_STEPS; k++) {
                let x = o + d * (a + (f32(k) + 0.5) * step);
                var bits = mask;
                while bits != 0u {
                    let v = firstTrailingBit(bits);
                    bits &= bits - 1u;
                    if media[v].flags.z == 1u {
                        depth += medium_density(v, x) * step;
                    }
                }
            }
        }
        if i == c.count || c.t[i] >= reach {
            break;
        }
        mask ^= 1u << c.volume[i];
        a = c.t[i];
    }
    return exp(-depth);
}
