// The two levels' traversals (bvh.rs builds them; its tests trace them on
// the CPU): the top level over the instances (common.wgsl's `instances`),
// each leaf's instances entered by taking the ray into its mesh's own space
// and walking that mesh's BVH from its root. An affine placement keeps the
// distance along the ray, so `t` means the same at both levels. Any hit
// (`occluded`), what light gets through (`transmittance`) and the closest
// hit (`closest`). Bound as group 1 by the ray-tracing passes.

struct Node {
    min: vec3<f32>,
    // First child (interior), or first triangle (a mesh's leaf) or first
    // instance (a top-level leaf).
    a: u32,
    max: vec3<f32>,
    // Triangles or instances in a leaf; 0 for an interior node.
    count: u32,
}

struct Triangle {
    // xyz: the first corner; the w's: the triangle's mean tint (rgb).
    v0: vec4<f32>,
    e1: vec4<f32>,
    e2: vec4<f32>,
}

@group(1) @binding(0) var<storage, read> nodes: array<Node>;
@group(1) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(1) @binding(2) var<storage, read> top_nodes: array<Node>;
// Bindings 3 to 5: the volumes' BVH (medium.wgsl); 6 the instances
// (common.wgsl); 7 the lights (light.wgsl).

// STACK (bvh::MAX_DEPTH + 1) comes from gpu/kernel.rs.

// A ray: its origin, direction and the direction's reciprocal.
struct Ray {
    o: vec3<f32>,
    d: vec3<f32>,
    inv: vec3<f32>,
}

fn ray(o: vec3<f32>, d: vec3<f32>) -> Ray {
    return Ray(o, d, 1.0 / d);
}

// `r` in instance `i`'s mesh's own space.
fn into(i: Instance, r: Ray) -> Ray {
    let o = vec4<f32>(r.o, 1.0);
    return ray(
        vec3<f32>(dot(i.to_own_x, o), dot(i.to_own_y, o), dot(i.to_own_z, o)),
        vec3<f32>(dot(i.to_own_x.xyz, r.d), dot(i.to_own_y.xyz, r.d), dot(i.to_own_z.xyz, r.d)),
    );
}

// Where `r` enters node `n`'s box within (0, t_max), or -1 when it misses.
fn box_entry(n: Node, r: Ray, t_max: f32) -> f32 {
    let t0 = (n.min - r.o) * r.inv;
    let t1 = (n.max - r.o) * r.inv;
    let lo = min(t0, t1);
    let hi = max(t0, t1);
    let near = max(max(max(lo.x, lo.y), lo.z), 0.0);
    let far = min(min(min(hi.x, hi.y), hi.z), t_max);
    return select(-1.0, near, near <= far);
}

// Möller–Trumbore: the distance along the ray to the triangle, or -1 when
// the ray misses it.
fn triangle_distance(tri: Triangle, r: Ray) -> f32 {
    let p = cross(r.d, tri.e2.xyz);
    let det = dot(tri.e1.xyz, p);
    if abs(det) < 1e-12 {
        return -1.0;
    }
    let inv = 1.0 / det;
    let s = r.o - tri.v0.xyz;
    let u = dot(s, p) * inv;
    if u < 0.0 || u > 1.0 {
        return -1.0;
    }
    let q = cross(s, tri.e1.xyz);
    let v = dot(r.d, q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return -1.0;
    }
    return dot(tri.e2.xyz, q) * inv;
}

// Whether the triangle lies along the ray in (0, t_max).
fn hits_triangle(tri: Triangle, r: Ray, t_max: f32) -> bool {
    let t = triangle_distance(tri, r);
    return t > 0.0 && t < t_max;
}

// Whether a triangle of the mesh whose BVH starts at `root` lies along `r`
// (its own space) within (0, t_max).
fn mesh_occluded(root: u32, r: Ray, t_max: f32) -> bool {
    var stack: array<u32, STACK>;
    var top = 1u;
    stack[0] = root;
    while top > 0u {
        top -= 1u;
        let n = nodes[stack[top]];
        if box_entry(n, r, t_max) < 0.0 {
            continue;
        }
        if n.count == 0u {
            stack[top] = n.a;
            stack[top + 1u] = n.a + 1u;
            top += 2u;
            continue;
        }
        for (var k = n.a; k < n.a + n.count; k++) {
            if hits_triangle(triangles[k], r, t_max) {
                return true;
            }
        }
    }
    return false;
}

// Whether anything lies along `o + t·d` with 0 < t < t_max.
fn occluded(o: vec3<f32>, d: vec3<f32>, t_max: f32) -> bool {
    let r = ray(o, d);
    var stack: array<u32, STACK>;
    var top = 1u;
    stack[0] = 0u;
    while top > 0u {
        top -= 1u;
        let n = top_nodes[stack[top]];
        if box_entry(n, r, t_max) < 0.0 {
            continue;
        }
        if n.count == 0u {
            stack[top] = n.a;
            stack[top + 1u] = n.a + 1u;
            top += 2u;
            continue;
        }
        for (var k = n.a; k < n.a + n.count; k++) {
            let i = instances[k];
            if mesh_occluded(i.id.y, into(i, r), t_max) {
                return true;
            }
        }
    }
    return false;
}

// Below this a shadow ray's light counts as gone.
const OPAQUE: f32 = 1e-3;

// What gets through `r` (own space) within (0, t_max) of the mesh whose BVH
// starts at `root`, a triangle of it letting `through` through, onto `kept`.
fn mesh_transmittance(root: u32, r: Ray, t_max: f32, through: vec3<f32>, kept: vec3<f32>) -> vec3<f32> {
    var left = kept;
    var stack: array<u32, STACK>;
    var top = 1u;
    stack[0] = root;
    while top > 0u {
        top -= 1u;
        let n = nodes[stack[top]];
        if box_entry(n, r, t_max) < 0.0 {
            continue;
        }
        if n.count == 0u {
            stack[top] = n.a;
            stack[top + 1u] = n.a + 1u;
            top += 2u;
            continue;
        }
        for (var k = n.a; k < n.a + n.count; k++) {
            if hits_triangle(triangles[k], r, t_max) {
                left *= through;
                if max(max(left.x, left.y), left.z) < OPAQUE {
                    return vec3<f32>(0.0);
                }
            }
        }
    }
    return left;
}

// What gets through along `o + t·d` with 0 < t < t_max: the product of
// the transmittances of the triangles there (a thin translucent surface
// lets translucency × translucency color through; an opaque one nothing).
// The order of the hits does not matter, so no sorting.
fn transmittance(o: vec3<f32>, d: vec3<f32>, t_max: f32) -> vec3<f32> {
    let r = ray(o, d);
    var through = vec3<f32>(1.0);
    var stack: array<u32, STACK>;
    var top = 1u;
    stack[0] = 0u;
    while top > 0u {
        top -= 1u;
        let n = top_nodes[stack[top]];
        if box_entry(n, r, t_max) < 0.0 {
            continue;
        }
        if n.count == 0u {
            stack[top] = n.a;
            stack[top + 1u] = n.a + 1u;
            top += 2u;
            continue;
        }
        for (var k = n.a; k < n.a + n.count; k++) {
            let i = instances[k];
            through = mesh_transmittance(i.id.y, into(i, r), t_max, i.transmit.rgb, through);
            if max(max(through.x, through.y), through.z) < OPAQUE {
                return vec3<f32>(0.0);
            }
        }
    }
    return through;
}

// A closest hit: the distance, the triangle and the instance; `t` is the
// limit asked for when nothing is hit.
struct Hit {
    t: f32,
    triangle: u32,
    instance: u32,
}

// `hit`, or the closest triangle of instance `k`'s mesh nearer along `r`
// (own space). Children are visited nearer first, so far boxes are mostly
// culled by the hit found.
fn mesh_closest(k: u32, r: Ray, hit: Hit) -> Hit {
    var best = hit;
    var stack: array<u32, STACK>;
    var top = 1u;
    stack[0] = instances[k].id.y;
    if box_entry(nodes[stack[0]], r, best.t) < 0.0 {
        return best;
    }
    while top > 0u {
        top -= 1u;
        let n = nodes[stack[top]];
        if n.count == 0u {
            let ta = box_entry(nodes[n.a], r, best.t);
            let tb = box_entry(nodes[n.a + 1u], r, best.t);
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
        for (var j = n.a; j < n.a + n.count; j++) {
            let t = triangle_distance(triangles[j], r);
            if t > 0.0 && t < best.t {
                best = Hit(t, j, k);
            }
        }
    }
    return best;
}

// The closest triangle along `o + t·d` with 0 < t < t_max.
fn closest(o: vec3<f32>, d: vec3<f32>, t_max: f32) -> Hit {
    let r = ray(o, d);
    var hit = Hit(t_max, 0u, 0u);
    var stack: array<u32, STACK>;
    var top = 1u;
    stack[0] = 0u;
    while top > 0u {
        top -= 1u;
        let n = top_nodes[stack[top]];
        if box_entry(n, r, hit.t) < 0.0 {
            continue;
        }
        if n.count == 0u {
            stack[top] = n.a;
            stack[top + 1u] = n.a + 1u;
            top += 2u;
            continue;
        }
        for (var k = n.a; k < n.a + n.count; k++) {
            hit = mesh_closest(k, into(instances[k], r), hit);
        }
    }
    return hit;
}

// The world normal (not unit) of `hit`'s triangle.
fn hit_normal(hit: Hit) -> vec3<f32> {
    let tri = triangles[hit.triangle];
    return own_to_world_normal(instances[hit.instance], cross(tri.e1.xyz, tri.e2.xyz));
}
