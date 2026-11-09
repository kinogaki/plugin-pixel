// The two levels' traversals (bvh.rs builds them; its tests trace them on
// the CPU): the top level over the instances (common.wgsl's `instances`),
// each leaf's instances entered by taking the ray into its mesh's own space
// and walking that mesh's BVH from its root. An affine placement keeps the
// distance along the ray, so `t` means the same at both levels. Any hit
// (`occluded`), what light gets through (`transmittance`) and the closest
// hit (`closest`). A triangle an instance's cutout cuts away where the ray
// meets it is no hit: the G-buffer's test (surface.wgsl), which comes
// before this part. Bound as group 1 by the ray-tracing passes.

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

// A traced triangle of a mesh its material cuts out (records.rs): what the
// G-buffer reads at its corners, in the triangle's order.
struct Cut {
    // Each corner's texture coordinates.
    s: array<f32, 3>,
    t: array<f32, 3>,
    // Each corner's alpha.
    alpha: array<f32, 3>,
}

@group(1) @binding(0) var<storage, read> nodes: array<Node>;
@group(1) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(1) @binding(2) var<storage, read> top_nodes: array<Node>;
// Bindings 3 to 5: the volumes' BVH (medium.wgsl); 6 the instances
// (common.wgsl); 7 the lights (light.wgsl); 9 the atlas (surface.wgsl).
@group(1) @binding(8) var<storage, read> cuts: array<Cut>;

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
// An empty box (min above max: the top level of a scene with no instances)
// is entered by no ray: the slab test alone would enter it everywhere, and
// walk the zeroed fallback instance's mesh without end.
fn box_entry(n: Node, r: Ray, t_max: f32) -> f32 {
    let t0 = (n.min - r.o) * r.inv;
    let t1 = (n.max - r.o) * r.inv;
    let lo = min(t0, t1);
    let hi = max(t0, t1);
    let near = max(max(max(lo.x, lo.y), lo.z), 0.0);
    let far = min(min(min(hi.x, hi.y), hi.z), t_max);
    return select(-1.0, near, near <= far && all(n.min <= n.max));
}

// Möller–Trumbore: where the ray crosses the triangle, (t, u, v): the
// distance along the ray and the second and third corners' weights there;
// t is -1 when the ray misses it.
fn crossing(tri: Triangle, r: Ray) -> vec3<f32> {
    let p = cross(r.d, tri.e2.xyz);
    let det = dot(tri.e1.xyz, p);
    if abs(det) < 1e-12 {
        return vec3<f32>(-1.0);
    }
    let inv = 1.0 / det;
    let s = r.o - tri.v0.xyz;
    let u = dot(s, p) * inv;
    if u < 0.0 || u > 1.0 {
        return vec3<f32>(-1.0);
    }
    let q = cross(s, tri.e1.xyz);
    let v = dot(r.d, q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return vec3<f32>(-1.0);
    }
    return vec3<f32>(dot(tri.e2.xyz, q) * inv, u, v);
}

// The three corners' values, as a vector.
fn per_corner(a: array<f32, 3>) -> vec3<f32> {
    return vec3<f32>(a[0], a[1], a[2]);
}

// Whether instance `k` cuts its traced triangle `j` away where the second
// and third corners weigh `w`: at the texture coordinates and alpha the
// corners blend to there. Only an instance with a cutout reads more than
// its flags (its record, and its mesh's cut records `id.w` places), so the
// loops calling this keep no more of an instance than its index.
fn cut_through(k: u32, j: u32, w: vec2<f32>) -> bool {
    if (instances[k].texture.w & CUTOUT) == 0u {
        return false;
    }
    let i = instances[k];
    let c = cuts[j - i.id.w];
    let b = vec3<f32>(1.0 - w.x - w.y, w.x, w.y);
    let uv = vec2<f32>(dot(b, per_corner(c.s)), dot(b, per_corner(c.t)));
    let alpha = dot(b, per_corner(c.alpha));
    return cut_away(i, surface_at(i, uv, vec4<f32>(1.0, 1.0, 1.0, alpha)).a);
}

// Whether instance `k`'s traced triangle `j` lies along `r` (its mesh's
// own space) within (0, t_max), where it isn't cut away.
fn hits(k: u32, j: u32, r: Ray, t_max: f32) -> bool {
    let x = crossing(triangles[j], r);
    return x.x > 0.0 && x.x < t_max && !cut_through(k, j, x.yz);
}

// Whether a triangle of instance `k`'s mesh lies along `r` (its own space)
// within (0, t_max).
fn mesh_occluded(k: u32, r: Ray, t_max: f32) -> bool {
    var stack: array<u32, STACK>;
    var top = 1u;
    stack[0] = instances[k].id.y;
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
        for (var j = n.a; j < n.a + n.count; j++) {
            if hits(k, j, r, t_max) {
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
            if mesh_occluded(k, into(instances[k], r), t_max) {
                return true;
            }
        }
    }
    return false;
}

// Below this a shadow ray's light counts as gone.
const OPAQUE: f32 = 1e-3;

// What gets through `r` (own space) within (0, t_max) of instance `k`'s
// mesh, a triangle of it letting its transmittance through, onto `kept`.
fn mesh_transmittance(k: u32, r: Ray, t_max: f32, kept: vec3<f32>) -> vec3<f32> {
    let through = instances[k].transmit.rgb;
    var left = kept;
    var stack: array<u32, STACK>;
    var top = 1u;
    stack[0] = instances[k].id.y;
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
        for (var j = n.a; j < n.a + n.count; j++) {
            if hits(k, j, r, t_max) {
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
// lets translucency × translucency color through; an opaque one nothing),
// of the instances casting the shadows of the light of slot `slot`. The
// order of the hits does not matter, so no sorting.
fn transmittance(o: vec3<f32>, d: vec3<f32>, t_max: f32, slot: u32) -> vec3<f32> {
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
            if !casts_for(instances[k], slot) {
                continue;
            }
            through = mesh_transmittance(k, into(instances[k], r), t_max, through);
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
            let x = crossing(triangles[j], r);
            if x.x > 0.0 && x.x < best.t && !cut_through(k, j, x.yz) {
                best = Hit(x.x, j, k);
            }
        }
    }
    return best;
}

// The closest triangle along `o + t·d` with 0 < t < t_max, of the
// instances casting the shadows of the light of slot `slot` (NO_SLOT: any).
fn closest(o: vec3<f32>, d: vec3<f32>, t_max: f32, slot: u32) -> Hit {
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
            if casts_for(instances[k], slot) {
                hit = mesh_closest(k, into(instances[k], r), hit);
            }
        }
    }
    return hit;
}

// The world normal (not unit) of `hit`'s triangle.
fn hit_normal(hit: Hit) -> vec3<f32> {
    let tri = triangles[hit.triangle];
    return own_to_world_normal(instances[hit.instance], cross(tri.e1.xyz, tri.e2.xyz));
}
