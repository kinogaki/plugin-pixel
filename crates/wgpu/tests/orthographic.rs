//! An orthographic camera on the GPU: a sphere's silhouette is the same
//! circle wherever it sits in the view (a perspective camera would stretch
//! it toward the edges), and the film grid stays locked to the view.
//! `cargo test -- --ignored` runs it.

mod support;

use kinogaki_pixel_render::{Aov, Camera, Projection, Scene};
use kinogaki_pixel_render_wgpu::{Renderer, View, Viewport};
use support::{canonical, device, look_at, sphere};

const W: u32 = 640;
const H: u32 = 360;

/// A ground-level orthographic camera 8 units across, looking down -Z.
fn camera() -> Camera {
    Camera {
        projection: Projection::Orthographic,
        horizontal_aperture: 8000.0,
        vertical_aperture: 4500.0,
        mm_per_unit: 1000.0,
        clipping_range: [0.1, 1e30],
        transform: look_at([0.0, 0.5, 10.0], [0.0, 0.5, 0.0]),
        ..Camera::default()
    }
}

/// The film pixels sphere id 9 covers: their count and bounding box.
fn silhouette(v: &Viewport) -> (usize, [u32; 4]) {
    let ids = v.read_aov(Aov::Id).expect("ids");
    let (w, h) = (ids.width, ids.height);
    let mut count = 0;
    let mut bounds = [u32::MAX, u32::MAX, 0, 0];
    for y in 0..h {
        for x in 0..w {
            if ids.ints()[(y * w + x) as usize] == 9 {
                count += 1;
                bounds = [
                    bounds[0].min(x),
                    bounds[1].min(y),
                    bounds[2].max(x),
                    bounds[3].max(y),
                ];
            }
        }
    }
    (count, bounds)
}

#[test]
#[ignore = "needs a GPU"]
fn an_orthographic_sphere_stays_a_circle_across_the_view() {
    let (device, queue) = device();
    let mut r = Renderer::new(&device, &queue);
    let mut v = Viewport::new(&r, 1);
    let view = View {
        camera: camera(),
        width: W,
        height: H,
    };
    // 0.5 units of radius at 80 view pixels a unit, 4 a film pixel: 10
    // film pixels.
    let radius_px = 0.5 * (W as f64 / 8.0) / 4.0;
    let mut first = None;
    for (x, y, z) in [
        (0.0, 0.5, 0.0),
        (-3.2, 1.6, 0.0),
        (3.1, -0.6, -4.0),
        (2.0, 1.0, 3.0),
    ] {
        let mut scene = Scene {
            meshes: vec![sphere([x, y, z], 0.5, 9, 0)],
            ..canonical()
        };
        scene.lights.clear();
        r.set_scene(&scene);
        r.render(&mut v, &view, 0).expect("a frame");
        let (count, b) = silhouette(&v);
        let width = (b[2] - b[0] + 1) as f64;
        let height = (b[3] - b[1] + 1) as f64;
        let area = std::f64::consts::PI * radius_px * radius_px;
        println!("sphere at ({x}, {y}, {z}): {count} px, {width}×{height}");
        // A circle of the radius, whatever its depth or place.
        assert!((width - 2.0 * radius_px).abs() <= 1.0, "width {width}");
        assert!((height - 2.0 * radius_px).abs() <= 1.0, "height {height}");
        assert!((count as f64 - area).abs() < 0.08 * area, "area {count}");
        // The same circle at every place (pixels on its rim may differ).
        let first = *first.get_or_insert(count);
        assert!(count.abs_diff(first) <= 8, "{count} against {first}");
    }
}

#[test]
#[ignore = "needs a GPU"]
fn an_orthographic_view_moves_by_whole_film_pixels_with_its_camera() {
    // The grid is locked to the view: a camera moved one film pixel's
    // width to the left shows the scene one film pixel to the right.
    let (device, queue) = device();
    let mut r = Renderer::new(&device, &queue);
    r.set_scene(&canonical());
    let mut v = Viewport::new(&r, 1);
    let mut view = View {
        camera: camera(),
        width: W,
        height: H,
    };
    r.render(&mut v, &view, 0).expect("a frame");
    let a = v.read_aov(Aov::Id).unwrap();
    let step = 8.0 / (W as f32 / 4.0);
    view.camera.transform[3][0] -= step;
    r.render(&mut v, &view, 0).expect("a frame");
    let b = v.read_aov(Aov::Id).unwrap();
    let (w, h) = (a.width, a.height);
    let mut same = 0;
    let mut all = 0;
    for y in 0..h {
        for x in 0..w - 1 {
            all += 1;
            same +=
                usize::from(a.ints()[(y * w + x) as usize] == b.ints()[(y * w + x + 1) as usize]);
        }
    }
    assert!(same as f64 > 0.995 * all as f64, "{same} of {all}");
}
