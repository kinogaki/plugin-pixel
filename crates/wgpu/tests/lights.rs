//! Distant lights, spots, gobos, a dome's ground color, rim lights and fog
//! on the GPU, against their definitions (the pixel crate's lights.rs,
//! gobo.rs and shading.rs). `cargo test -- --ignored` runs them.

mod support;

use kinogaki_pixel_render::{
    Aov, AovImage, Camera, Dome, Fog, Gobo, Light, LightShape, PixelSurface, Projection, Rims,
    Scene, Settings,
};
use kinogaki_pixel_render_wgpu::{Renderer, View, Viewport};
use support::{device, look_at, plane, preview, sphere, write_png};

const W: u32 = 256;
const H: u32 = 256;
const FILM: u32 = 64;
const PI: f32 = std::f32::consts::PI;

/// An orthographic camera looking straight down at the origin, 16 units
/// across.
fn top_view() -> View {
    View {
        camera: Camera {
            projection: Projection::Orthographic,
            horizontal_aperture: 16_000.0,
            vertical_aperture: 16_000.0,
            clipping_range: [0.1, 1e4],
            transform: look_at([0.0, 20.0, 0.01], [0.0, 0.0, 0.0]),
            ..Camera::default()
        },
        width: W,
        height: H,
    }
}

/// A white plane under `lights`, no dome.
fn plane_scene(lights: Vec<Light>) -> Scene {
    Scene {
        volumes: Vec::new(),
        meshes: vec![plane(40.0, 1, 0)],
        materials: vec![PixelSurface {
            color: [1.0; 3],
            ..PixelSurface::default()
        }],
        lights,
        dome: None,
        rims: Default::default(),
        fog: None,
    }
}

/// The default settings: with no dome, direct light alone.
fn direct_only() -> Settings {
    Settings::default()
}

fn render(scene: &Scene, settings: Settings, view: &View) -> Viewport {
    let (device, queue) = device();
    let mut r = Renderer::new(&device, &queue);
    r.set_settings(settings);
    r.set_scene(scene);
    let mut v = Viewport::new(&r, 1);
    r.render(&mut v, view, 0).expect("a frame");
    v
}

/// Film pixel (x, y)'s first channel.
fn red(image: &AovImage, x: u32, y: u32) -> f32 {
    image.floats()[((y * FILM + x) * 4) as usize]
}

/// A distant light along `direction`'s -Z from `from` toward the origin.
fn sun(from: [f64; 3], intensity: f32, angle: f32) -> Light {
    Light {
        shape: LightShape::Distant { angle },
        intensity,
        normalize: true,
        transform: look_at(from, [0.0, 0.0, 0.0]),
        ..Light::default()
    }
}

#[test]
#[ignore = "needs a GPU"]
fn distant_light_is_parallel_and_casts_parallel_shadows() {
    // From (0, 1, 2): cos 1/√5 on the ground, illuminance 3.
    let mut scene = plane_scene(vec![sun([0.0, 1.0, 2.0], 3.0, 0.53)]);
    // A small sphere 1 up at x = -4 shadows the ground 2 units toward -z.
    scene.meshes.push(sphere([-4.0, 1.0, 0.0], 0.5, 2, 0));
    let v = render(&scene, direct_only(), &top_view());
    let direct = v.read_aov(Aov::Direct).unwrap();
    let want = 3.0 / 5f32.sqrt() / PI;
    // The same everywhere on the open ground: no falloff.
    for (x, y) in [(8, 8), (56, 8), (32, 32), (56, 56)] {
        let got = red(&direct, x, y);
        assert!((got - want).abs() < want * 2e-3, "({x}, {y}): {got} {want}");
    }
    // The shadow: the light's line through the sphere's centre meets the
    // ground 2 units along -z of it (film y up is -z).
    let film = |x: f32, z: f32| (((x + 8.0) * 4.0) as u32, ((z + 8.0) * 4.0) as u32);
    let (sx, sy) = film(-4.0, -2.0);
    let ids = v.read_aov(Aov::Id).unwrap();
    assert_eq!(ids.ints()[(sy * FILM + sx) as usize], 1, "the ground");
    assert_eq!(red(&direct, sx, sy), 0.0, "in the shadow");
}

#[test]
#[ignore = "needs a GPU"]
fn spot_lights_fall_off_with_distance_and_their_cone() {
    // A spot 4 up looking straight down (tilted a hair), a 20° cone.
    let spot = Light {
        shape: LightShape::Spot {
            radius: 0.1,
            cone_angle: 20.0,
            cone_softness: 0.0,
        },
        intensity: 100.0,
        transform: look_at([0.0, 4.0, 0.0001], [0.0, 0.0, 0.0]),
        ..Light::default()
    };
    let v = render(&plane_scene(vec![spot]), direct_only(), &top_view());
    let direct = v.read_aov(Aov::Direct).unwrap();
    // Radiant intensity 100 × π r²; at the centre 4 below, cos 1.
    let intensity = 100.0 * PI * 0.01;
    let centre = red(&direct, 32, 32);
    // The film pixel's centre sits ⅛ unit off the axis.
    let want = intensity / 16.0 / PI;
    assert!((centre - want).abs() < want * 0.01, "{centre} {want}");
    // 4 units out is 45° off the axis: outside the cone, dark.
    assert_eq!(red(&direct, 48, 32), 0.0);
    // About a unit out (16°) is inside it: I cos / d².
    let at = |p: u32| -8.0 + (p as f32 + 0.5) / 4.0;
    let d2 = 16.0 + at(36) * at(36) + at(32) * at(32);
    let inside = red(&direct, 36, 32);
    let want = intensity * (4.0 / d2.sqrt()) / d2 / PI;
    assert!((inside - want).abs() < want * 0.01, "{inside} {want}");
}

#[test]
#[ignore = "needs a GPU"]
fn gobos_dapple_direct_light_by_their_seed() {
    let dappled = |seed| {
        let mut light = sun([0.0, 2.0, 1.0], 3.0, 0.0);
        light.gobo = Some(Gobo {
            scale: 1.5,
            contrast: 1.0,
            seed,
            offset: [0.0; 2],
        });
        let v = render(&plane_scene(vec![light]), direct_only(), &top_view());
        v.read_aov(Aov::Direct).unwrap()
    };
    let a = dappled(1);
    let full = 3.0 * (2.0 / 5f32.sqrt()) / PI;
    let (mut lit, mut dark) = (0, 0);
    for y in 0..FILM {
        for x in 0..FILM {
            let v = red(&a, x, y);
            lit += usize::from(v > full * 0.95);
            dark += usize::from(v < full * 0.05);
        }
    }
    let n = (FILM * FILM) as usize;
    assert!(lit > n / 6 && dark > n / 6, "lit {lit} dark {dark} of {n}");
    let b = dappled(2);
    let changed = (0..FILM * FILM)
        .filter(|&i| (a.floats()[i as usize * 4] - b.floats()[i as usize * 4]).abs() > full * 0.5)
        .count();
    assert!(changed > n / 8, "another seed, another pattern: {changed}");
    write_png("gobo_direct", FILM, FILM, preview(Aov::Direct, &a));
}

#[test]
#[ignore = "needs a GPU"]
fn a_domes_ground_color_lights_by_how_far_a_surface_faces_up() {
    let mut scene = plane_scene(Vec::new());
    scene.meshes.push(sphere([0.0, 2.0, 0.0], 1.0, 2, 0));
    // A dome of intensity ½: sky above the horizon, its ground color below.
    let (sky, ground) = ([0.4, 0.5, 0.8], [0.2, 0.1, 0.0]);
    scene.dome = Some(Dome {
        above: sky.map(|c| c * 0.5),
        below: ground.map(|c| c * 0.5),
        visible: true,
    });
    let settings = Settings {
        ao_distance: 0.5,
        ..Settings::default()
    };
    // From the side, level with the sphere's centre.
    let side = View {
        camera: Camera {
            projection: Projection::Orthographic,
            horizontal_aperture: 8000.0,
            vertical_aperture: 8000.0,
            clipping_range: [0.1, 1e4],
            transform: look_at([0.0, 2.0, 20.0], [0.0, 2.0, 0.0]),
            ..Camera::default()
        },
        width: W,
        height: H,
    };
    let v = render(&scene, settings, &side);
    let ambient = v.read_aov(Aov::Ambient).unwrap();
    let ao = v.read_aov(Aov::Ao).unwrap();
    // On the sphere (8 film pixels a unit, its centre at film (32, 32)),
    // the light is the ground's and the sky's mixed by n·up × ½ + ½, × ½,
    // × AO.
    let mut checked = 0;
    for y in 25..40u32 {
        for x in [30u32, 33] {
            let (wx, wy) = ((x as f32 + 0.5 - 32.0) / 8.0, (32.0 - y as f32 - 0.5) / 8.0);
            if wx * wx + wy * wy > 0.8 {
                continue;
            }
            let up = wy * 0.5 + 0.5;
            let i = ((y * FILM + x) * 4) as usize;
            let occlusion = ao.floats()[(y * FILM + x) as usize];
            for k in 0..3 {
                let want = (ground[k] + (sky[k] - ground[k]) * up) * 0.5 * occlusion;
                let got = ambient.floats()[i + k];
                assert!((got - want).abs() < 0.02, "({x}, {y}) {k}: {got} {want}");
            }
            checked += 1;
        }
    }
    assert!(checked > 15, "{checked}");
    // The background is the dome as the camera sees it: its sky looking
    // up past the horizon, its ground looking down.
    let empty = Scene {
        meshes: Vec::new(),
        ..scene.clone()
    };
    for (from, want) in [([0.0, -5.0, 20.0], sky), ([0.0, 9.0, 20.0], ground)] {
        let mut view = side;
        view.camera.transform = look_at(from, [0.0, 2.0, 0.0]);
        let beauty = render(&empty, settings, &view)
            .read_aov(Aov::Beauty)
            .unwrap();
        let got = &beauty.floats()[..4];
        for k in 0..3 {
            assert!((got[k] - want[k] * 0.5).abs() < 1e-3, "{got:?} {want:?}");
        }
        assert_eq!(got[3], 1.0);
    }
    // No dome, no ambient light.
    let dark = render(&plane_scene(Vec::new()), settings, &side);
    let ambient = dark.read_aov(Aov::Ambient).unwrap();
    assert!(ambient.floats().iter().all(|&v| v == 0.0 || v == 1.0));
}

#[test]
#[ignore = "needs a GPU"]
fn fog_covers_distance_and_posterizes() {
    // A long ground seen low from one end, lit evenly from above.
    let scene = plane_scene(vec![sun([0.0, 1.0, 0.2], 3.0, 0.0)]);
    let view = View {
        camera: Camera {
            focal_length: 20.0,
            transform: look_at([0.0, 1.0, 19.0], [0.0, 0.0, 0.0]),
            ..Camera::default()
        },
        width: W,
        height: H,
    };
    let fog = [0.3, 0.32, 0.4];
    let foggy = Scene {
        fog: Some(Fog {
            density: 0.4,
            color: fog,
            start: 2.0,
            falloff: 0.0,
            ..Fog::default()
        }),
        ..scene.clone()
    };
    let clear = render(&scene, direct_only(), &view);
    let v = render(&foggy, direct_only(), &view);
    let beauty = v.read_aov(Aov::Beauty).unwrap();
    let ids = v.read_aov(Aov::Id).unwrap();
    write_png("fog_beauty", FILM, FILM, preview(Aov::Beauty, &beauty));
    // The far ground (high in the film) is all fog; the near (low) the
    // clear render's.
    let px = |img: &AovImage, x: u32, y: u32| {
        let i = ((y * FILM + x) * 4) as usize;
        [0, 1, 2].map(|k| img.floats()[i + k])
    };
    let far = (0..FILM)
        .find(|&y| ids.ints()[(y * FILM + 32) as usize] == 1)
        .unwrap();
    let got = px(&beauty, 32, far);
    assert!((0..3).all(|k| (got[k] - fog[k]).abs() < 2e-3), "{got:?}");
    let clear_beauty = clear.read_aov(Aov::Beauty).unwrap();
    assert_eq!(px(&beauty, 32, FILM - 1), px(&clear_beauty, 32, FILM - 1));
    // Between, each pixel is the clear color mixed toward the fog by one
    // of the volumes' levels' amounts (Volume Levels: fog bands like the
    // volumes).
    let steps = (Settings::default().volume_levels - 1) as f32;
    for y in far..FILM {
        let (a, b) = (px(&clear_beauty, 32, y), px(&beauty, 32, y));
        let f = (b[0] - a[0]) / (fog[0] - a[0]);
        assert!(
            ((f * steps) - (f * steps).round()).abs() < 0.05,
            "row {y}: fog {f}"
        );
    }
}

#[test]
#[ignore = "needs a GPU"]
fn rims_brighten_silhouettes_where_lit() {
    let mut scene = plane_scene(Vec::new());
    scene.meshes[0].visible = false;
    scene.meshes.push(sphere([0.0, 0.0, 0.0], 3.0, 2, 0));
    let rim = |rims, lights| {
        let mut scene = scene.clone();
        scene.lights = lights;
        scene.rims = rims;
        render(&scene, direct_only(), &top_view())
            .read_aov(Aov::Beauty)
            .unwrap()
    };
    let white = |all: f32, lit: f32| Rims {
        all: [all; 3],
        lit: [lit; 3],
    };
    // Unlit, a rim on all sides lights the edge and not the middle.
    let all = rim(white(4.0, 0.0), Vec::new());
    let edge = red(&all, 32, 20);
    assert!(edge > red(&all, 32, 32) + 0.5, "{edge}");
    // On the lit side only: nothing without light.
    let lit = rim(white(0.0, 4.0), Vec::new());
    assert_eq!(red(&lit, 32, 20), red(&lit, 32, 32));
    // Lit, the Lit rims join in; rims add up.
    let sun = vec![sun([0.0, 20.0, 0.01], 0.05, 0.0)];
    let one = rim(white(0.0, 2.0), sun.clone());
    let both = rim(white(2.0, 2.0), sun.clone());
    let none = rim(white(0.0, 0.0), sun);
    assert!(red(&one, 32, 20) > red(&none, 32, 20), "a lit rim");
    assert!(red(&both, 32, 20) > red(&one, 32, 20), "two rims");
}

#[test]
#[ignore = "needs a GPU"]
fn authored_normals_shade_and_a_card_lit_from_behind_does_not_shadow_itself() {
    // A one-sided card seen from above, the sun below it at (0, -1, 2):
    // its face turns away from the sun. Shading normals leaning toward the
    // sun (Crown Normal's, two-sided foliage) light it by their cosine,
    // and its own plane does not shadow it.
    let lean = [0.0f32, 0.2, 1.0];
    let l = (lean[0] * lean[0] + lean[1] * lean[1] + lean[2] * lean[2]).sqrt();
    let n = lean.map(|c| c / l);
    let mut scene = plane_scene(vec![sun([0.0, -1.0, 2.0], 3.0, 0.53)]);
    scene.meshes[0].normals = Some(std::sync::Arc::from(vec![n; 6]));
    let v = render(&scene, direct_only(), &top_view());
    let direct = v.read_aov(Aov::Direct).unwrap();
    let wi = [0.0, -1.0 / 5f32.sqrt(), 2.0 / 5f32.sqrt()];
    let want = 3.0 * (n[1] * wi[1] + n[2] * wi[2]) / PI;
    for (x, y) in [(8, 8), (32, 32), (56, 56)] {
        let got = red(&direct, x, y);
        assert!((got - want).abs() < want * 2e-3, "({x}, {y}): {got} {want}");
    }
    // Its own flat normal faces away from the sun: no light.
    let flat = render(
        &plane_scene(vec![sun([0.0, -1.0, 2.0], 3.0, 0.53)]),
        direct_only(),
        &top_view(),
    );
    assert_eq!(red(&flat.read_aov(Aov::Direct).unwrap(), 32, 32), 0.0);
}
