//! Translucent Pixel Surfaces on the GPU: light on a thin surface's back
//! face comes through × translucency × translucency color, shadowed from
//! the back; opaque surfaces stay dark there. Shadow rays keep what thin
//! surfaces let through, so a translucent card's shadow is light × its
//! transmittance and two cards' the product. `cargo test -- --ignored`
//! runs them.

mod support;

use kinogaki_pixel_render::{Aov, PixelSurface, Scene, Settings, dither};
use support::{ortho, preview, quad, render, rgb, sun, write_png};

const FILM: u32 = 64;
const PI: f32 = std::f32::consts::PI;

/// A white 4×4 quad facing the camera (+z) with `surface`'s translucency,
/// and a dark blocker behind its right half (between it and a light
/// behind), lit by `light_from`.
fn scene(translucency: f32, light_from: [f64; 3]) -> Scene {
    let leaf = PixelSurface {
        color: [1.0; 3],
        translucency,
        translucency_color: [1.0, 0.5, 0.25],
        ..PixelSurface::default()
    };
    let h = 2.0;
    Scene {
        volumes: Vec::new(),
        meshes: vec![
            quad(
                [[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]],
                1,
                0,
            ),
            quad(
                [
                    [0.5, -5.0, -1.0],
                    [3.0, -5.0, -1.0],
                    [3.0, 5.0, -1.0],
                    [0.5, 5.0, -1.0],
                ],
                2,
                1,
            ),
        ],
        materials: vec![
            leaf,
            PixelSurface {
                color: [0.2; 3],
                ..PixelSurface::default()
            },
        ],
        lights: vec![sun(light_from, 3.0)],
        dome: None,
        rims: Default::default(),
        fog: None,
    }
}

/// Direct light alone (no dome: no ambient), no bounce.
fn direct_only() -> Settings {
    Settings {
        gi: 0.0,
        ..Settings::default()
    }
}

/// Film pixel (x, y) of a 64² film looking at the quad from +z, 8 across:
/// world x = -1 is column 24, x = 1 column 40.
fn at(x: u32, y: u32) -> usize {
    (y * FILM + x) as usize
}

#[test]
#[ignore = "needs a GPU"]
fn a_back_lit_thin_quad_glows_through_when_translucent() {
    let view = ortho([0.0, 0.0, 10.0], [0.0; 3], 8.0, FILM * 4);
    // The sun behind: from (0, 1, -2), cosine 2/√5 on the back face.
    let behind = [0.0, 1.0, -2.0];
    let lit = render(&scene(0.5, behind), direct_only(), &view);
    let dark = render(&scene(0.0, behind), direct_only(), &view);
    let (on, off) = (
        lit.read_aov(Aov::Direct).unwrap(),
        dark.read_aov(Aov::Direct).unwrap(),
    );
    // Through the left half: 3 × 2/√5 / π × translucency × its color.
    let e = 3.0 * 2.0 / 5f32.sqrt() / PI;
    let want = [0.5, 0.25, 0.125].map(|t| e * t);
    let got = rgb(&on, at(24, 32));
    assert!(
        (0..3).all(|k| (got[k] - want[k]).abs() < want[k] * 3e-3),
        "{got:?} {want:?}"
    );
    assert_eq!(rgb(&off, at(24, 32)), [0.0; 3], "opaque: dark");
    // The blocker shades the right half's back: dark either way.
    assert_eq!(rgb(&on, at(40, 32)), [0.0; 3], "shadowed from the back");
    // The beauty glows where light comes through.
    let y = |v: &kinogaki_pixel_render_wgpu::Viewport| {
        dither::luminance(rgb(&v.read_aov(Aov::Beauty).unwrap(), at(24, 32)))
    };
    assert!(y(&lit) > y(&dark) + 0.05, "{} {}", y(&lit), y(&dark));
    let beauty = lit.read_aov(Aov::Beauty).unwrap();
    write_png(
        "translucency_beauty",
        FILM,
        FILM,
        preview(Aov::Beauty, &beauty),
    );
}

#[test]
#[ignore = "needs a GPU"]
fn front_light_ignores_translucency() {
    let view = ortho([0.0, 0.0, 10.0], [0.0; 3], 8.0, FILM * 4);
    let front = [0.0, 1.0, 2.0];
    let a = render(&scene(0.8, front), direct_only(), &view);
    let b = render(&scene(0.0, front), direct_only(), &view);
    let (a, b) = (
        a.read_aov(Aov::Direct).unwrap(),
        b.read_aov(Aov::Direct).unwrap(),
    );
    // The quad itself (x and y −2 to 2: columns and rows 16 to 47) is lit
    // the same; the blocker behind it takes what it lets through.
    for y in 16..48 {
        for x in 16..48 {
            assert_eq!(rgb(&a, at(x, y)), rgb(&b, at(x, y)), "({x}, {y})");
        }
    }
}

/// A white floor at z 0 under one or two cards (at z 1 and 1.5, x from −4
/// to −2) of `translucency` each, in the same translucency color, lit by
/// a sun from −x at 45°: the cards' shadows fall on the floor from x −3
/// to −1, beside the cards as the camera above sees them.
fn cards(translucency: f32, two: bool) -> Scene {
    let card = |z: f32, id: u32| {
        quad(
            [
                [-4.0, -3.0, z],
                [-2.0, -3.0, z],
                [-2.0, 3.0, z],
                [-4.0, 3.0, z],
            ],
            id,
            0,
        )
    };
    let mut meshes = vec![
        quad(
            [
                [-9.0, -9.0, 0.0],
                [9.0, -9.0, 0.0],
                [9.0, 9.0, 0.0],
                [-9.0, 9.0, 0.0],
            ],
            1,
            1,
        ),
        card(1.0, 2),
    ];
    if two {
        meshes.push(card(1.5, 3));
    }
    Scene {
        volumes: Vec::new(),
        meshes,
        materials: vec![
            PixelSurface {
                color: [1.0; 3],
                translucency,
                translucency_color: [1.0, 0.5, 0.25],
                ..PixelSurface::default()
            },
            PixelSurface::default(),
        ],
        lights: vec![sun([-1.0, 0.0, 1.0], 3.0)],
        dome: None,
        rims: Default::default(),
        fog: None,
    }
}

#[test]
#[ignore = "needs a GPU"]
fn shadows_keep_what_thin_translucent_surfaces_let_through() {
    let view = ortho([0.0, 0.0, 10.0], [0.0; 3], 8.0, FILM * 4);
    let direct = |scene: &Scene| {
        render(scene, direct_only(), &view)
            .read_aov(Aov::Direct)
            .unwrap()
    };
    // The floor in the shadow (x −1.5, column 20) and in the open (x 2,
    // column 48): 3 × cos 45° / π in the open.
    let (shade, open) = (at(20, 32), at(48, 32));
    let e = 3.0 / 2f32.sqrt() / PI;
    let near = |got: [f32; 3], want: [f32; 3]| {
        (0..3).all(|k| (got[k] - want[k]).abs() <= want[k].max(1e-3) * 3e-3)
    };
    let one = direct(&cards(0.5, false));
    assert!(near(rgb(&one, open), [e; 3]), "{:?}", rgb(&one, open));
    let want = [0.5, 0.25, 0.125].map(|t| e * t);
    assert!(
        near(rgb(&one, shade), want),
        "{:?} {want:?}",
        rgb(&one, shade)
    );
    // Two cards: the product.
    let two = direct(&cards(0.5, true));
    let want = [0.25, 0.0625, 0.015625].map(|t| e * t);
    assert!(
        near(rgb(&two, shade), want),
        "{:?} {want:?}",
        rgb(&two, shade)
    );
    // An opaque card casts a black shadow.
    let opaque = direct(&cards(0.0, false));
    assert_eq!(rgb(&opaque, shade), [0.0; 3]);
}
