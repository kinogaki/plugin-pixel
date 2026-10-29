//! Frame times at viewport sizes: GPU time per pass (timestamps, which
//! overlap where passes run side by side: see `PassTimes`) and the wall time
//! of a submitted frame, waited on. Prints a table; asserts the 1080p
//! canonical frame at pixel size 4 stays well under a 60 Hz frame.
//! `cargo test --release --test timing -- --ignored --nocapture`.

mod support;

use std::time::Instant;

use kinogaki_pixel_render::{Fog, PixelVolume, Scene, Settings, Volume};
use kinogaki_pixel_render_wgpu::{PASSES, Renderer, Viewport};
use support::{canonical, canonical_view, cuboid, device, sphere};

/// The canonical scene plus a grid of 64 more spheres (≈265k triangles).
fn heavy() -> Scene {
    let mut scene = canonical();
    for i in 0..64 {
        let (x, z) = ((i % 8) as f32 * 1.6 - 5.6, (i / 8) as f32 * -1.6 - 2.0);
        scene.meshes.push(sphere([x, 0.6, z], 0.6, 10 + i, 0));
    }
    scene
}

/// The heavy scene in low noisy mist (a volume box over the ground),
/// marched with shadow rays (and, below, lit fog).
fn misty() -> Scene {
    let mut scene = heavy();
    scene.volumes.push(Volume {
        mesh: cuboid([-12.0, 0.0, -16.0], [12.0, 1.5, 6.0], 99),
        medium: PixelVolume {
            density: 0.4,
            noise_amount: 0.8,
            edge_falloff: 0.5,
            ..PixelVolume::default()
        },
    });
    scene
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Median GPU ms per pass and in total, and median wall ms a frame.
fn measure(r: &mut Renderer, v: &mut Viewport, width: u32, height: u32) -> (Vec<f64>, f64, f64) {
    let view = canonical_view(width, height);
    for _ in 0..20 {
        r.render(v, &view, 0);
        v.read_timings();
    }
    let mut passes = vec![Vec::new(); PASSES.len()];
    let mut totals = Vec::new();
    let mut walls = Vec::new();
    for _ in 0..200 {
        let start = Instant::now();
        r.render(v, &view, 0);
        let times = v.read_timings().expect("timestamps");
        walls.push(start.elapsed().as_secs_f64() * 1e3);
        for (k, (_, ms)) in times.passes.iter().enumerate() {
            passes[k].push(*ms);
        }
        totals.push(times.total());
    }
    (
        passes.iter_mut().map(|p| median(p)).collect(),
        median(&mut totals),
        median(&mut walls),
    )
}

#[test]
#[ignore = "needs a GPU; timings"]
fn frame_times() {
    let (device, queue) = device();
    let mut r = Renderer::new(&device, &queue);
    let mut v = Viewport::new(&r, 1);
    if !v.timed() {
        eprintln!("no TIMESTAMP_QUERY on this device; skipped");
        return;
    }
    println!(
        "{:<10} {:>9} {:>4} {:>9} | {} | {:>7} {:>7}",
        "scene",
        "view",
        "px",
        "film",
        PASSES.map(|p| format!("{p:>7}")).join(" "),
        "gpu ms",
        "wall ms"
    );
    let mut canonical_1080p4 = 0.0;
    for (name, scene) in [
        ("canonical", canonical()),
        ("heavy", heavy()),
        ("misty", misty()),
    ] {
        r.set_scene(&scene);
        for (w, h, px) in [
            (1920, 1080, 4),
            (1920, 1080, 2),
            (1920, 1080, 1),
            (3840, 2160, 4),
        ] {
            // The misty scene's fog is lit too.
            let lit = name == "misty";
            r.set_settings(Settings {
                pixel_size: px,
                ..Settings::default()
            });
            r.set_fog(lit.then(|| Fog {
                density: 0.02,
                inscatter: true,
                ..Fog::default()
            }));
            let (passes, total, wall) = measure(&mut r, &mut v, w, h);
            let film = v.film().unwrap().size;
            println!(
                "{name:<10} {:>9} {px:>4} {:>9} | {} | {total:>7.3} {wall:>7.3}",
                format!("{w}x{h}"),
                format!("{}x{}", film[0], film[1]),
                passes
                    .iter()
                    .map(|p| format!("{p:>7.3}"))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            if name == "canonical" && (w, px) == (1920, 4) {
                canonical_1080p4 = wall;
            }
        }
    }
    assert!(
        canonical_1080p4 < 8.0,
        "1080p at pixel size 4: {canonical_1080p4} ms"
    );
}
