//! One diffuse bounce on the GPU: a white floor beside a sunlit red wall
//! takes red light off it, as a CPU trace of the same fixed ray pattern
//! says; the bounce is deterministic, and GI 0 traces nothing. `cargo test
//! -- --ignored` runs them.

mod support;

use std::sync::Arc;

use kinogaki_pixel_render::geometry::Geometry;
use kinogaki_pixel_render::{Aov, PixelSurface, Scene, Settings, hemisphere};
use support::{ortho, plane, preview, quad, render, rgb, sun, write_png};

const FILM: u32 = 64;
const PI: f32 = std::f32::consts::PI;
/// The sun's illuminance and where it shines from.
const SUN: f32 = 3.0;
const SUN_FROM: [f64; 3] = [2.0, 1.5, 0.5];
/// The wall's material color and its corners' tint (Cd).
const WALL: [f32; 3] = [1.0, 1.0, 0.5];
const TINT: [f32; 3] = [0.8, 0.05, 0.1];

/// A white floor and a wall standing on it along z at x = 0, facing +x,
/// colored by its material × its tint.
fn scene() -> Scene {
    let mut wall = quad(
        [
            [0.0, 0.0, 6.0],
            [0.0, 0.0, -6.0],
            [0.0, 3.0, -6.0],
            [0.0, 3.0, 6.0],
        ],
        2,
        1,
    );
    wall.tints = Some(Arc::from(vec![TINT; 6]));
    Scene {
        volumes: Vec::new(),
        meshes: vec![plane(40.0, 1, 0), wall],
        materials: vec![
            PixelSurface {
                color: [1.0; 3],
                ..PixelSurface::default()
            },
            PixelSurface {
                color: WALL,
                ..PixelSurface::default()
            },
        ],
        lights: vec![sun(SUN_FROM, SUN)],
        dome: None,
        rims: Default::default(),
        fog: None,
    }
}

fn settings(gi: f32) -> Settings {
    Settings {
        gi,
        gi_samples: 16,
        gi_distance: 4.0,
        ..Settings::default()
    }
}

/// Looking down at x ∈ [-1, 7], z ∈ [-4, 4]: 8 film pixels a unit.
fn top() -> kinogaki_pixel_render_wgpu::View {
    ortho([3.0, 20.0, 0.001], [3.0, 0.0, 0.0], 8.0, FILM * 4)
}

/// Film pixel (col, row)'s point on the floor.
fn floor_point(col: u32, row: u32) -> [f32; 3] {
    let f = |p: u32| (p as f32 + 0.5) / 8.0;
    [-1.0 + f(col), 0.0, -4.0 + f(row)]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The bounce at floor point `p` traced on the CPU: the same 16 rays, each
/// hit lit by the sun (Lambert, shadowed) × its albedo, their mean / π.
fn reference(geometry: &Geometry, materials: &[[f32; 3]], p: [f32; 3]) -> [f32; 3] {
    let n = [0.0, 1.0, 0.0];
    let l = SUN_FROM.map(|v| v as f32);
    let len = dot(l, l).sqrt();
    let wi = l.map(|v| v / len);
    let origin = [p[0], 1e-4, p[2]];
    let mut sum = [0.0f32; 3];
    for k in 0..16 {
        let d = hemisphere::around(n, k, 16);
        let Some(hit) = geometry.bvh.intersect(origin, d, 4.0) else {
            continue;
        };
        let v = &geometry.vertices[hit.triangle as usize * 3..][..3];
        let (a, b, c) = (v[0].position, v[1].position, v[2].position);
        let e1 = [0, 1, 2].map(|i| b[i] - a[i]);
        let e2 = [0, 1, 2].map(|i| c[i] - a[i]);
        let mut hn = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let m = dot(hn, hn).sqrt();
        hn = hn.map(|x| x / m);
        if dot(hn, d) > 0.0 {
            hn = hn.map(|x| -x);
        }
        let x = [0, 1, 2].map(|i| origin[i] + d[i] * hit.t + hn[i] * 1e-4);
        let cos = dot(hn, wi);
        if cos <= 0.0 || geometry.bvh.occluded(x, wi, 1e30) {
            continue;
        }
        let (material, _) = geometry.objects[v[0].object as usize];
        for k in 0..3 {
            sum[k] += SUN * cos * materials[material as usize][k] * v[0].tint[k];
        }
    }
    sum.map(|s| s / (16.0 * PI))
}

#[test]
#[ignore = "needs a GPU"]
fn a_floor_beside_a_lit_red_wall_takes_red_bounce_light() {
    let scene = scene();
    let v = render(&scene, settings(1.0), &top());
    let gi = v.read_aov(Aov::Indirect).unwrap();
    let ids = v.read_aov(Aov::Id).unwrap();
    write_png("bounce_indirect", FILM, FILM, preview(Aov::Indirect, &gi));
    let geometry = Geometry::build(&scene.meshes);
    let materials = [[1.0; 3], WALL];
    let (mut checked, mut differ) = (0, 0);
    for row in 8..56 {
        // Columns from just right of the wall (x = 0 is column 8) out to
        // x = 3.
        for col in 9..32 {
            let i = (row * FILM + col) as usize;
            assert_eq!(ids.ints()[i], 1, "the floor at ({col}, {row})");
            let want = reference(&geometry, &materials, floor_point(col, row));
            let got = rgb(&gi, i);
            checked += 1;
            if (0..3).any(|k| (got[k] - want[k]).abs() > want[k] * 5e-3 + 1e-4) {
                differ += 1;
            }
        }
    }
    // A ray grazing the wall's edge may hit or miss in either's float.
    assert!(differ * 100 <= checked, "{differ} of {checked} differ");
    // Red off the wall: near it the bounce is red; out past GI Distance
    // nothing is in reach, exactly.
    let near = rgb(&gi, (32 * FILM + 10) as usize);
    assert!(near[0] > 0.05 && near[0] > 5.0 * near[1], "{near:?}");
    assert_eq!(rgb(&gi, (32 * FILM + 60) as usize), [0.0; 3]);
}

#[test]
#[ignore = "needs a GPU"]
fn the_bounce_is_deterministic() {
    let scene = scene();
    let a = render(&scene, settings(1.0), &top());
    let b = render(&scene, settings(1.0), &top());
    let gi = a.read_aov(Aov::Indirect).unwrap();
    assert_eq!(gi, b.read_aov(Aov::Indirect).unwrap());
    assert_eq!(
        a.read_aov(Aov::Beauty).unwrap(),
        b.read_aov(Aov::Beauty).unwrap()
    );
}

#[test]
#[ignore = "needs a GPU"]
fn gi_zero_traces_nothing() {
    let scene = scene();
    let off = render(&scene, settings(0.0), &top());
    let gi = off.read_aov(Aov::Indirect).unwrap();
    assert!(gi.floats().chunks(4).all(|p| p[..3] == [0.0; 3]));
    // Neither its samples nor its distance reach the image.
    let other = Settings {
        gi_samples: 1,
        gi_distance: 100.0,
        ..settings(0.0)
    };
    let again = render(&scene, other, &top());
    assert_eq!(
        off.read_aov(Aov::Beauty).unwrap(),
        again.read_aov(Aov::Beauty).unwrap()
    );
    // With GI the floor by the wall brightens.
    let on = render(&scene, settings(1.0), &top());
    let i = (32 * FILM + 10) as usize;
    let lum = |v: &kinogaki_pixel_render_wgpu::Viewport| {
        kinogaki_pixel_render::dither::luminance(rgb(&v.read_aov(Aov::Direct).unwrap(), i))
            + kinogaki_pixel_render::dither::luminance(rgb(&v.read_aov(Aov::Indirect).unwrap(), i))
    };
    assert!(lum(&on) > lum(&off));
}
