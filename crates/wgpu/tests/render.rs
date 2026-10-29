//! The canonical scene on the GPU: its AOVs as PNGs, the shading checked
//! against the CPU definition, and the upscale checked pixel for pixel.
//! `cargo test -- --ignored` runs them.

mod support;

use kinogaki_pixel_render::{Aov, Ramp, Settings, dither, shading};
use kinogaki_pixel_render_wgpu::{Renderer, Viewport};
use support::{PLANE, SPHERE, canonical, canonical_view, device, preview, write_png};

fn rendered() -> (Renderer, Viewport) {
    rendered_with(Settings::default())
}

fn rendered_with(settings: Settings) -> (Renderer, Viewport) {
    let (device, queue) = device();
    let mut r = Renderer::new(&device, &queue);
    r.set_settings(settings);
    r.set_scene(&canonical());
    let mut v = Viewport::new(&r, 1);
    r.render(&mut v, &canonical_view(640, 360), 0)
        .expect("a frame");
    (r, v)
}

#[test]
#[ignore = "needs a GPU"]
fn canonical_scene_writes_its_aovs() {
    let (_r, v) = rendered();
    let film = v.film().expect("a film");
    assert_eq!(film.size, [160, 90]);
    for aov in Aov::ALL {
        let image = v.read_aov(aov).expect("the output");
        assert_eq!((image.width, image.height), (160, 90));
        let path = write_png(aov.name(), image.width, image.height, preview(aov, &image));
        println!("{}", path.display());
    }
    let view = v.read_image(0).expect("the view image");
    let path = write_png("view", view.width, view.height, preview(Aov::Beauty, &view));
    println!("{}", path.display());

    let ids = v.read_aov(Aov::Id).expect("ids");
    let ids = ids.ints();
    let at = |x: u32, y: u32| ids[(y * 160 + x) as usize];
    // The sphere fills the middle; the plane the bottom corners; the dome
    // the top corners.
    assert_eq!(at(80, 40), SPHERE);
    assert_eq!(at(5, 85), PLANE);
    assert_eq!(at(2, 2), u32::MAX);

    // The sphere's lit side (toward the light, right and up) is brighter
    // than its shadowed side.
    let direct = v.read_aov(Aov::Direct).expect("direct");
    let d = |x: u32, y: u32| direct.floats()[((y * 160 + x) * 4) as usize];
    assert!(d(92, 30) > 0.2, "lit side {}", d(92, 30));
    assert_eq!(d(66, 50), 0.0, "shadowed side");
    // The sphere casts a shadow on the plane (down-left of it, away from
    // the light), where the plane is otherwise lit.
    let shadowed = (0..160)
        .flat_map(|x| (0..90).map(move |y| (x, y)))
        .filter(|&(x, y)| at(x, y) == PLANE && d(x, y) == 0.0)
        .count();
    assert!(shadowed > 100, "shadow pixels {shadowed}");

    // AO darkens the plane where the sphere touches it.
    let ao = v.read_aov(Aov::Ao).expect("ao");
    let a = |x: u32, y: u32| ao.floats()[(y * 160 + x) as usize];
    let contact = (60..100)
        .flat_map(|x| (60..80).map(move |y| (x, y)))
        .filter(|&(x, y)| at(x, y) == PLANE)
        .map(|(x, y)| a(x, y))
        .fold(1.0f32, f32::min);
    assert!(contact < 0.6, "contact AO {contact}");
    assert!(a(5, 85) > 0.95, "open plane AO {}", a(5, 85));
}

#[test]
#[ignore = "needs a GPU"]
fn shading_matches_the_cpu_definition() {
    // The shade pass is shading::shade over the same inputs (no rim, no
    // fog): recompute the beauty from the albedo, direct, ambient and
    // indirect outputs.
    let (r, v) = rendered();
    let s = r.settings();
    let table = dither::thresholds(s.dither);
    let albedo = v.read_aov(Aov::Albedo).unwrap();
    let direct = v.read_aov(Aov::Direct).unwrap();
    let ambient = v.read_aov(Aov::Ambient).unwrap();
    let indirect = v.read_aov(Aov::Indirect).unwrap();
    let beauty = v.read_aov(Aov::Beauty).unwrap();
    let ids = v.read_aov(Aov::Id).unwrap();
    let mut differ = 0;
    let mut covered = 0;
    for y in 0..90u32 {
        for x in 0..160u32 {
            let i = (y * 160 + x) as usize;
            if ids.ints()[i] == u32::MAX {
                continue;
            }
            covered += 1;
            let c = |img: &kinogaki_pixel_render::AovImage| {
                let f = img.floats();
                [f[i * 4], f[i * 4 + 1], f[i * 4 + 2]]
            };
            let (al, di, am, gi) = (c(&albedo), c(&direct), c(&ambient), c(&indirect));
            let light = [0, 1, 2].map(|k| di[k] + am[k] + gi[k]);
            let t = dither::threshold_at(&table, s.dither.size(), x, y);
            let shaded = shading::shade(light, &s.shade_ramp, s.shade_levels, t, s.dither_strength);
            let want = [0, 1, 2].map(|k| al[k] * shaded[k]);
            let got = c(&beauty);
            // Half-float storage of the beauty: 2⁻¹⁰ relative.
            if (0..3).any(|k| (want[k] - got[k]).abs() > want[k].abs() * 2e-3 + 1e-4) {
                differ += 1;
            }
        }
    }
    // A light exactly on a step's cut may round either way in f32.
    assert!(covered > 5000);
    assert!(differ * 1000 <= covered, "{differ} of {covered} differ");
}

#[test]
#[ignore = "needs a GPU"]
fn shaded_light_sits_on_the_levels() {
    // Through a grey ramp, beauty / albedo is the quantized light, whose
    // luminance is one of the levels k / (shade_levels - 1).
    let (r, v) = rendered_with(Settings {
        shade_ramp: Ramp::grey(),
        ..Settings::default()
    });
    let steps = (r.settings().shade_levels - 1) as f32;
    let albedo = v.read_aov(Aov::Albedo).unwrap();
    let beauty = v.read_aov(Aov::Beauty).unwrap();
    let ids = v.read_aov(Aov::Id).unwrap();
    let mut seen = [false; 8];
    for i in 0..(160 * 90) {
        if ids.ints()[i] == u32::MAX {
            continue;
        }
        let (a, b) = (albedo.floats(), beauty.floats());
        let light = [0, 1, 2].map(|k| b[i * 4 + k] / a[i * 4 + k]);
        let level = dither::luminance(light) * steps;
        assert!(
            (level - level.round()).abs() < 0.01,
            "pixel {i}: level {level}"
        );
        seen[(level.round() as usize).min(7)] = true;
    }
    // The scene spans dark to lit.
    assert!(seen.iter().filter(|&&s| s).count() >= 3, "{seen:?}");
}

#[test]
#[ignore = "needs a GPU"]
fn the_view_is_the_film_upscaled_exactly() {
    let (device, queue) = device();
    let mut r = Renderer::new(&device, &queue);
    r.set_scene(&canonical());
    let mut v = Viewport::new(&r, 1);
    // A view that is not a multiple of the pixel size.
    r.render(&mut v, &canonical_view(643, 358), 0)
        .expect("a frame");
    let film = v.film().unwrap();
    assert_eq!(film.size, [161, 90]);
    let beauty = v.read_aov(Aov::Beauty).unwrap();
    let view = v.read_image(0).unwrap();
    assert_eq!((view.width, view.height), (643, 358));
    for y in 0..358u32 {
        for x in 0..643u32 {
            let [i, j] = film.film_pixel(x, y);
            let v = ((y * 643 + x) * 4) as usize;
            let f = ((j * 161 + i) * 4) as usize;
            assert_eq!(
                view.floats()[v..v + 4],
                beauty.floats()[f..f + 4],
                "({x}, {y})"
            );
        }
    }
}
