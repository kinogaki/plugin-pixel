//! Pixel Volumes and lit fog on the GPU, against closed forms: a constant
//! slab's transmittance is Beer–Lambert's, its light from a distant light
//! is single scattering's, an occluder cuts a shaft, a shadowing volume
//! dims the ground, lit fog matches its own closed form, the bounds are
//! never drawn, frames repeat exactly, and the shade pass bands volumes on
//! their own levels as `shading::over` does. `cargo test -- --ignored`
//! runs them.

mod support;

use kinogaki_pixel_render::shading::{Bands, over};
use kinogaki_pixel_render::volume::phase;
use kinogaki_pixel_render::{
    Aov, AovImage, Fog, PixelSurface, PixelVolume, Scene, Settings, Volume, dither,
};
use support::{cuboid, ortho, plane, preview, quad, render, rgb, sun, write_png};

const FILM: u32 = 32;
/// The slab's thickness along the view (z from −1 to 1).
const THICK: f32 = 2.0;

/// A slab 6 wide and high, `THICK` deep, of `medium`, seen from +z.
fn slab(medium: PixelVolume) -> Volume {
    Volume {
        mesh: cuboid([-3.0, -3.0, -1.0], [3.0, 3.0, 1.0], 9),
        medium,
    }
}

/// Light from the lights and emission alone (no dome: no ambient), no
/// bounce.
fn bare() -> Settings {
    Settings {
        gi: 0.0,
        ..Settings::default()
    }
}

fn centre() -> usize {
    ((FILM / 2) * FILM + FILM / 2) as usize
}

fn volume_at(image: &AovImage, i: usize) -> [f32; 4] {
    let f = image.floats();
    [f[i * 4], f[i * 4 + 1], f[i * 4 + 2], f[i * 4 + 3]]
}

fn view() -> kinogaki_pixel_render_wgpu::View {
    ortho([0.0, 0.0, 10.0], [0.0; 3], 4.0, FILM * 4)
}

#[test]
#[ignore = "needs a GPU"]
fn a_constant_slab_keeps_beer_lambert() {
    for density in [0.1, 0.5, 2.0] {
        let scene = Scene {
            volumes: vec![slab(PixelVolume {
                density,
                ..PixelVolume::default()
            })],
            ..Scene::default()
        };
        let v = render(&scene, bare(), &view())
            .read_aov(Aov::Volume)
            .unwrap();
        let [r, g, b, a] = volume_at(&v, centre());
        let want = 1.0 - (-density * THICK).exp();
        assert!((a - want).abs() < 2e-3, "density {density}: {a} {want}");
        assert_eq!([r, g, b], [0.0; 3], "nothing lights it");
    }
}

#[test]
#[ignore = "needs a GPU"]
fn a_camera_inside_a_volume_sees_from_where_it_stands() {
    // The camera's plane (z = 10) is inside a box reaching to z = −1.
    let density = 0.1;
    let scene = Scene {
        volumes: vec![Volume {
            mesh: cuboid([-3.0, -3.0, -1.0], [3.0, 3.0, 30.0], 9),
            medium: PixelVolume {
                density,
                ..PixelVolume::default()
            },
        }],
        ..Scene::default()
    };
    let v = render(&scene, bare(), &view())
        .read_aov(Aov::Volume)
        .unwrap();
    let want = 1.0 - (-density * 11.0).exp();
    assert!((volume_at(&v, centre())[3] - want).abs() < 2e-3);
}

#[test]
#[ignore = "needs a GPU"]
fn a_distant_light_scatters_as_single_scattering_says() {
    let (density, albedo, g) = (0.4, [0.9, 0.6, 0.3], 0.5);
    // A sun up and behind the slab: the camera looks toward it a little.
    let from = [0.0, 3.0, -4.0];
    let scene = Scene {
        volumes: vec![slab(PixelVolume {
            density,
            scatter_color: albedo,
            anisotropy: g,
            ..PixelVolume::default()
        })],
        lights: vec![sun(from.map(f64::from), 2.0)],
        ..Scene::default()
    };
    let v = render(&scene, bare(), &view())
        .read_aov(Aov::Volume)
        .unwrap();
    let got = volume_at(&v, centre());
    // L = albedo · p(g, cos θ) · E · (1 − e^(−σ L)), cos θ = wi · d.
    let cos = 4.0 / 5.0;
    let opacity = 1.0 - (-density * THICK).exp();
    for k in 0..3 {
        let want = albedo[k] * phase(g, cos) * 2.0 * opacity;
        assert!((got[k] - want).abs() < want * 4e-3, "{k}: {got:?} {want}");
    }
    // Emission alone: ∫ T σ e dt = e (1 − T), e its emission × color.
    let glowing = Scene {
        volumes: vec![slab(PixelVolume {
            density,
            emission: 0.5,
            emission_color: [1.0, 0.5, 0.0],
            ..PixelVolume::default()
        })],
        ..Scene::default()
    };
    let v = render(&glowing, bare(), &view())
        .read_aov(Aov::Volume)
        .unwrap();
    let e = volume_at(&v, centre());
    assert!((e[0] - 0.5 * opacity).abs() < 2e-3 && (e[1] - 0.25 * opacity).abs() < 2e-3);
}

/// The slab under a sun straight above, half shaded by a roof over x < 0.
fn shaft(shadow_rays: bool) -> AovImage {
    let roof = quad(
        [
            [-8.0, 4.0, -8.0],
            [0.0, 4.0, -8.0],
            [0.0, 4.0, 8.0],
            [-8.0, 4.0, 8.0],
        ],
        1,
        0,
    );
    let scene = Scene {
        meshes: vec![roof],
        materials: vec![PixelSurface::default()],
        volumes: vec![slab(PixelVolume::default())],
        lights: vec![sun([0.0, 10.0, 0.001], 3.0)],
        ..Scene::default()
    };
    let settings = Settings {
        volume_shadow_rays: shadow_rays,
        ..bare()
    };
    render(&scene, settings, &view())
        .read_aov(Aov::Volume)
        .unwrap()
}

#[test]
#[ignore = "needs a GPU"]
fn an_occluder_casts_a_shaft() {
    let v = shaft(true);
    let row = FILM / 2;
    let (left, right) = ((row * FILM + 4) as usize, (row * FILM + FILM - 5) as usize);
    let (l, r) = (rgb(&v, left)[1], rgb(&v, right)[1]);
    assert!(r > 0.01, "lit: {r}");
    assert!(l < r * 1e-3, "in the shaft's shadow: {l} vs {r}");
    // Without shadow rays both sides are lit alike.
    let open = shaft(false);
    let (l, r) = (rgb(&open, left)[1], rgb(&open, right)[1]);
    assert!((l - r).abs() < r * 1e-3, "{l} {r}");
    write_png("volume_shaft", FILM, FILM, preview(Aov::Volume, &v));
}

#[test]
#[ignore = "needs a GPU"]
fn a_shadowing_volume_dims_the_ground() {
    // A box of fog 2 high over part of the ground, sun straight above.
    let ground = |shadows| Scene {
        meshes: vec![plane(40.0, 2, 0)],
        materials: vec![PixelSurface::default()],
        volumes: vec![Volume {
            mesh: cuboid([0.0, 1.0, -9.0], [9.0, 3.0, 9.0], 9),
            medium: PixelVolume {
                density: 0.5,
                shadows,
                ..PixelVolume::default()
            },
        }],
        lights: vec![sun([0.0, 10.0, 0.001], 3.0)],
        ..Scene::default()
    };
    let top = ortho([0.0, 20.0, 0.0001], [0.0; 3], 8.0, FILM * 4);
    let direct = |shadows| {
        render(&ground(shadows), bare(), &top)
            .read_aov(Aov::Direct)
            .unwrap()
    };
    let (dim, clear) = (direct(true), direct(false));
    let (under, beside) = (
        ((FILM / 2) * FILM + FILM - 4) as usize,
        ((FILM / 2) * FILM + 3) as usize,
    );
    let e = 3.0 / std::f32::consts::PI;
    let kept = (-0.5f32 * 2.0).exp();
    assert!((rgb(&clear, under)[0] - e).abs() < e * 1e-3);
    assert!(
        (rgb(&dim, under)[0] - e * kept).abs() < e * 3e-3,
        "{:?}",
        rgb(&dim, under)
    );
    assert!((rgb(&dim, beside)[0] - e).abs() < e * 1e-3);
}

#[test]
#[ignore = "needs a GPU"]
fn lit_fog_matches_its_closed_form() {
    let (density, reach) = (0.05, 30.0);
    let fog = Fog {
        density,
        color: [0.5, 0.6, 0.7],
        start: 0.0,
        height: 0.0,
        falloff: 0.0,
        inscatter: true,
        anisotropy: 0.4,
    };
    let scene = Scene {
        lights: vec![sun([0.0, 3.0, -4.0], 2.0)],
        fog: Some(fog),
        ..Scene::default()
    };
    let settings = Settings {
        volume_max_distance: reach,
        ..bare()
    };
    let v = render(&scene, settings, &view())
        .read_aov(Aov::Volume)
        .unwrap();
    let got = volume_at(&v, centre());
    let opacity = 1.0 - (-density * reach).exp();
    assert!((got[3] - opacity).abs() < 2e-3, "{got:?} {opacity}");
    let want = 0.6 * phase(0.4, 0.8) * 2.0 * opacity;
    assert!((got[1] - want).abs() < want * 4e-3, "{got:?} {want}");
    // Unlit, the same fog writes no volume: the shade pass hazes it.
    let flat = Scene {
        fog: Some(Fog {
            inscatter: false,
            ..fog
        }),
        ..scene
    };
    let v = render(&flat, settings, &view())
        .read_aov(Aov::Volume)
        .unwrap();
    assert_eq!(volume_at(&v, centre()), [0.0; 4]);
}

#[test]
#[ignore = "needs a GPU"]
fn the_bounds_are_not_drawn_and_frames_repeat() {
    let scene = Scene {
        meshes: vec![quad(
            [
                [-9.0, -9.0, -3.0],
                [9.0, -9.0, -3.0],
                [9.0, 9.0, -3.0],
                [-9.0, 9.0, -3.0],
            ],
            4,
            0,
        )],
        materials: vec![PixelSurface::default()],
        volumes: vec![slab(PixelVolume {
            noise_amount: 1.0,
            noise_frequency: 1.5,
            edge_falloff: 0.5,
            ..PixelVolume::default()
        })],
        lights: vec![sun([1.0, 3.0, 2.0], 3.0)],
        ..Scene::default()
    };
    let a = render(&scene, Settings::default(), &view());
    // The wall behind shows through: its id and depth, not the slab's.
    let id = a.read_aov(Aov::Id).unwrap();
    assert!(id.ints().iter().all(|&i| i == 4), "the slab is not drawn");
    let depth = a.read_aov(Aov::Depth).unwrap();
    assert!((depth.floats()[centre()] - 13.0).abs() < 1e-3);
    let b = render(&scene, Settings::default(), &view());
    for aov in [Aov::Volume, Aov::Beauty] {
        assert_eq!(a.read_aov(aov), b.read_aov(aov), "{aov:?} repeats");
    }
    // Noise and edges shape it: not every pixel alike.
    let v = a.read_aov(Aov::Volume).unwrap();
    let opacities: Vec<f32> = (0..(FILM * FILM) as usize)
        .map(|i| volume_at(&v, i)[3])
        .collect();
    let (lo, hi) = opacities
        .iter()
        .fold((1.0f32, 0.0f32), |(l, h), &o| (l.min(o), h.max(o)));
    assert!(hi - lo > 0.2, "{lo} {hi}");
    write_png(
        "volume_beauty",
        FILM,
        FILM,
        preview(Aov::Beauty, &a.read_aov(Aov::Beauty).unwrap()),
    );
}

#[test]
#[ignore = "needs a GPU"]
fn the_shade_pass_bands_volumes_as_over_says() {
    // A noisy, glowing slab against the empty background: its opacity
    // ramps across many bands.
    let scene = Scene {
        volumes: vec![slab(PixelVolume {
            density: 0.6,
            emission: 0.6,
            noise_amount: 1.0,
            noise_frequency: 0.8,
            edge_falloff: 1.0,
            ..PixelVolume::default()
        })],
        ..Scene::default()
    };
    let settings = Settings {
        volume_levels: 6,
        volume_dither: 0.3,
        ..bare()
    };
    let a = render(&scene, settings, &view());
    let volume = a.read_aov(Aov::Volume).unwrap();
    let beauty = a.read_aov(Aov::Beauty).unwrap();
    let table = dither::thresholds(settings.dither);
    let bands = Bands::of(&settings);
    let (mut differ, mut seen) = (0, 0);
    for y in 0..FILM {
        for x in 0..FILM {
            let i = (y * FILM + x) as usize;
            let [r, g, b, opacity] = volume_at(&volume, i);
            seen += usize::from(opacity > 1e-4);
            let t = dither::threshold_at(&table, settings.dither.size(), x, y);
            let want = over(
                [0.0; 3],
                ([r, g, b], opacity),
                &settings.shade_ramp,
                bands,
                t,
            );
            let got = rgb(&beauty, i);
            // Half-float storage of the beauty: 2⁻¹⁰ relative.
            if (0..3).any(|k| (want[k] - got[k]).abs() > want[k].abs() * 2e-3 + 1e-4) {
                differ += 1;
            }
        }
    }
    assert!(
        seen > (FILM * FILM / 2) as usize,
        "the slab covers the film"
    );
    // An opacity exactly on a cut may round either way in f32.
    assert!(differ * 100 <= seen, "{differ} of {seen} differ");
    let b = render(&scene, settings, &view());
    assert_eq!(b.read_aov(Aov::Beauty).unwrap(), beauty, "frames repeat");
    write_png("volume_bands", FILM, FILM, preview(Aov::Beauty, &beauty));
}
