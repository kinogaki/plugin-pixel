// Any-hit and closest-hit traversals of the BVH (bvh.rs builds it;
// Bvh::occluded and Bvh::intersect are these on the CPU), and what light
// gets through along a ray (Bvh::transmittance). Bound as group 1 by the
// ray-tracing passes.

struct Node {
    min: vec3<f32>,
    // First child (interior) or first triangle (leaf).
    a: u32,
    max: vec3<f32>,
    // Triangles in a leaf; 0 for an interior node.
    count: u32,
}

struct Triangle {
    v0: vec4<f32>,
    e1: vec4<f32>,
    e2: vec4<f32>,
}

// What a ray hitting a stored triangle shades with (geometry::Surface).
struct Surface {
    // The mean of its corners' tints.
    tint: vec3<f32>,
    // Index into the object table.
    object: u32,
}

@group(1) @binding(0) var<storage, read> nodes: array<Node>;
@group(1) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(1) @binding(2) var<storage, read> surfaces: array<Surface>;
// Bindings 3 to 5: the volumes' BVH (medium.wgsl).
@group(1) @binding(6) var<storage, read> objects: array<Object>;

// STACK (bvh::MAX_DEPTH + 1) comes from gpu/layout.rs.

fn hits_box(n: Node, o: vec3<f32>, inv: vec3<f32>, t_max: f32) -> bool {
    let t0 = (n.min - o) * inv;
    let t1 = (n.max - o) * inv;
    let lo = min(t0, t1);
    let hi = max(t0, t1);
    let near = max(max(max(lo.x, lo.y), lo.z), 0.0);
    let far = min(min(min(hi.x, hi.y), hi.z), t_max);
    return near <= far;
}

// Möller–Trumbore: the distance along the ray to the triangle, or -1 when
// the ray misses it.
fn triangle_distance(tri: Triangle, o: vec3<f32>, d: vec3<f32>) -> f32 {
    let p = cross(d, tri.e2.xyz);
    let det = dot(tri.e1.xyz, p);
    if abs(det) < 1e-12 {
        return -1.0;
    }
    let inv = 1.0 / det;
    let s = o - tri.v0.xyz;
    let u = dot(s, p) * inv;
    if u < 0.0 || u > 1.0 {
        return -1.0;
    }
    let q = cross(s, tri.e1.xyz);
    let v = dot(d, q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return -1.0;
    }
    return dot(tri.e2.xyz, q) * inv;
}

// Whether the triangle lies along the ray in (0, t_max).
fn hits_triangle(tri: Triangle, o: vec3<f32>, d: vec3<f32>, t_max: f32) -> bool {
    let t = triangle_distance(tri, o, d);
    return t > 0.0 && t < t_max;
}

// Whether anything lies along `o + t·d` with 0 < t < t_max.
fn occluded(o: vec3<f32>, d: vec3<f32>, t_max: f32) -> bool {
    let inv = 1.0 / d;
    var stack: array<u32, STACK>;
    var top = 1u;
    stack[0] = 0u;
    while top > 0u {
        top -= 1u;
        let n = nodes[stack[top]];
        if !hits_box(n, o, inv, t_max) {
            continue;
        }
        if n.count == 0u {
            stack[top] = n.a;
            stack[top + 1u] = n.a + 1u;
            top += 2u;
            continue;
        }
        for (var k = n.a; k < n.a + n.count; k++) {
            if hits_triangle(triangles[k], o, d, t_max) {
                return true;
            }
        }
    }
    return false;
}

// Below this a shadow ray's light counts as gone.
const OPAQUE: f32 = 1e-3;

// What gets through along `o + t·d` with 0 < t < t_max: the product of
// the transmittances of the triangles there (a thin translucent surface
// lets translucency × translucency color through; an opaque one nothing).
// The order of the hits does not matter, so no sorting.
fn transmittance(o: vec3<f32>, d: vec3<f32>, t_max: f32) -> vec3<f32> {
    let inv = 1.0 / d;
    var through = vec3<f32>(1.0);
    var stack: array<u32, STACK>;
    var top = 1u;
    stack[0] = 0u;
    while top > 0u {
        top -= 1u;
        let n = nodes[stack[top]];
        if !hits_box(n, o, inv, t_max) {
            continue;
        }
        if n.count == 0u {
            stack[top] = n.a;
            stack[top + 1u] = n.a + 1u;
            top += 2u;
            continue;
        }
        for (var k = n.a; k < n.a + n.count; k++) {
            if hits_triangle(triangles[k], o, d, t_max) {
                through *= objects[surfaces[k].object].transmit.rgb;
                if max(max(through.x, through.y), through.z) < OPAQUE {
                    return vec3<f32>(0.0);
                }
            }
        }
    }
    return through;
}

// A closest hit: the distance and the stored triangle's index; `t` is the
// limit asked for when nothing is hit.
struct Hit {
    t: f32,
    triangle: u32,
}

// Where the ray enters node `n`'s box within (0, t_max), or -1 when it
// misses it.
fn box_entry(n: Node, o: vec3<f32>, inv: vec3<f32>, t_max: f32) -> f32 {
    let t0 = (n.min - o) * inv;
    let t1 = (n.max - o) * inv;
    let lo = min(t0, t1);
    let hi = max(t0, t1);
    let near = max(max(max(lo.x, lo.y), lo.z), 0.0);
    let far = min(min(min(hi.x, hi.y), hi.z), t_max);
    return select(-1.0, near, near <= far);
}

// The closest triangle along `o + t·d` with 0 < t < t_max. Children are
// visited nearer first, so far boxes are mostly culled by the hit found.
fn closest(o: vec3<f32>, d: vec3<f32>, t_max: f32) -> Hit {
    let inv = 1.0 / d;
    var hit = Hit(t_max, 0u);
    if box_entry(nodes[0], o, inv, t_max) < 0.0 {
        return hit;
    }
    var stack: array<u32, STACK>;
    var top = 1u;
    stack[0] = 0u;
    while top > 0u {
        top -= 1u;
        let n = nodes[stack[top]];
        if n.count == 0u {
            let ta = box_entry(nodes[n.a], o, inv, hit.t);
            let tb = box_entry(nodes[n.a + 1u], o, inv, hit.t);
            // Push the farther first, so the nearer pops next.
            let near_first = ta <= tb;
            let far_t = select(ta, tb, near_first);
            let near_t = select(tb, ta, near_first);
            if far_t >= 0.0 {
                stack[top] = select(n.a, n.a + 1u, near_first);
                top += 1u;
            }
            if near_t >= 0.0 {
                stack[top] = select(n.a + 1u, n.a, near_first);
                top += 1u;
            }
            continue;
        }
        for (var k = n.a; k < n.a + n.count; k++) {
            let t = triangle_distance(triangles[k], o, d);
            if t > 0.0 && t < hit.t {
                hit = Hit(t, k);
            }
        }
    }
    return hit;
}
