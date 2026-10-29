//! Temporal stability: a still camera renders identically, and a sub-pixel
//! camera move changes AO and the dithered beauty only as much as the
//! moved surface implies (no per-frame noise, no swimming pattern).

mod support;

use kinogaki_pixel_render::camera::CameraFrame;
use kinogaki_pixel_render::{Aov, Film, dither};
use kinogaki_pixel_render_wgpu::{Renderer, View, Viewport};
use support::{canonical, canonical_view, device};

const W: u32 = 640;
const H: u32 = 360;

/// A frame's AO, shade level a pixel (the quantized light's luminance in
/// steps: beauty / albedo) and ids.
fn frame(r: &mut Renderer, v: &mut Viewport, view: &View) -> (Vec<f32>, Vec<i32>, Vec<u32>) {
    r.render(v, view, 0).expect("a frame");
    let steps = (r.settings().shade_levels - 1) as f32;
    let ao = v.read_aov(Aov::Ao).unwrap().floats().to_vec();
    let beauty = v.read_aov(Aov::Beauty).unwrap();
    let albedo = v.read_aov(Aov::Albedo).unwrap();
    let levels = beauty
        .floats()
        .chunks(4)
        .zip(albedo.floats().chunks(4))
        .map(|(b, a)| {
            let light = [0, 1, 2].map(|k| if a[k] > 0.0 { b[k] / a[k] } else { 0.0 });
            (dither::luminance(light) * steps).round() as i32
        })
        .collect();
    let ids = v.read_aov(Aov::Id).unwrap().ints().to_vec();
    (ao, levels, ids)
}

/// `view` with its camera moved `pixels` film pixels to the right, at the
/// sphere's depth (about 7 units).
fn moved(view: &View, pixels: f64) -> View {
    let film = Film::new(view.width, view.height, 4);
    let cam = CameraFrame::new(&view.camera, &film, None);
    let shift = cam.axes[0] * (pixels * cam.step * 7.0);
    let mut out = *view;
    for k in 0..3 {
        out.camera.transform[3][k] += shift[k] as f32;
    }
    out
}

#[test]
#[ignore = "needs a GPU"]
fn a_still_camera_renders_identically() {
    let (device, queue) = device();
    let mut r = Renderer::new(&device, &queue);
    r.set_scene(&canonical());
    let mut v = Viewport::new(&r, 1);
    let view = canonical_view(W, H);
    let a = frame(&mut r, &mut v, &view);
    for _ in 0..3 {
        assert!(frame(&mut r, &mut v, &view) == a, "frames differ");
    }
    // Rendering never rebuilt the geometry.
    assert_eq!(r.geometry_builds(), 1);
}

#[test]
#[ignore = "needs a GPU"]
fn sub_pixel_moves_keep_ao_and_dither_still() {
    let (device, queue) = device();
    let mut r = Renderer::new(&device, &queue);
    r.set_scene(&canonical());
    let mut v = Viewport::new(&r, 1);
    let view = canonical_view(W, H);
    let (fw, fh) = (W as usize / 4, H as usize / 4);
    let (ao0, levels0, ids0) = frame(&mut r, &mut v, &view);
    for step in [0.1, 0.25, 0.4] {
        let (ao1, levels1, ids1) = frame(&mut r, &mut v, &moved(&view, step));
        // Interior pixels: the same object over their 3×3 neighbourhood in
        // both frames (edges legitimately change as the silhouette moves).
        let mut deltas = Vec::new();
        let mut changed = 0usize;
        for y in 1..fh - 1 {
            for x in 1..fw - 1 {
                let id = ids0[y * fw + x];
                let interior = id != u32::MAX
                    && (0..9).all(|k| {
                        let i = (y + k / 3 - 1) * fw + x + k % 3 - 1;
                        ids0[i] == id && ids1[i] == id
                    });
                if !interior {
                    continue;
                }
                let i = y * fw + x;
                deltas.push((ao1[i] - ao0[i]).abs());
                if levels1[i] != levels0[i] {
                    changed += 1;
                }
            }
        }
        deltas.sort_by(f32::total_cmp);
        let n = deltas.len();
        let mean = deltas.iter().sum::<f32>() / n as f32;
        let p99 = deltas[n * 99 / 100];
        let fraction = changed as f32 / n as f32;
        println!(
            "move {step} px: {n} interior pixels, AO mean |Δ| {mean:.4}, p99 {p99:.4}, max {:.4}; shade level changed {:.2}%",
            deltas[n - 1],
            fraction * 100.0
        );
        // One AO ray flipping is 1/32.
        assert!(mean < 0.01, "AO mean change {mean}");
        assert!(p99 <= 3.0 / 32.0 + 1e-6, "AO p99 change {p99}");
        // The dither pattern stays on the film grid: only pixels whose light
        // crosses a threshold change.
        assert!(
            fraction < 0.03,
            "shade level changed on {fraction} of pixels"
        );
    }
}
