//! How long a moved object takes to reach the GPU beside a large Instancer:
//! the scene sync and the upload, measured by hand
//! (`cargo test --release --test gpu -- --ignored hand_run --nocapture`).
use kinogaki_pixel::gpu::Renderer;
use kinogaki_pixel::sync::Sync;
use kinogaki_render_api::display::wgpu;
use kinogaki_render_api::geometry::{Column, Sampled, Vec3, Xform};
use kinogaki_render_api::{Change, Dirty, Instancer, Prim, PrimId, SceneUpdate, Shape, Topology};
use kinogaki_studio_api_testing::engine;
use std::time::Instant;

fn id(slot: u32) -> PrimId {
    PrimId {
        slot,
        generation: 0,
    }
}

/// A grid of `n` × `n` quads (2n² triangles) in the XZ plane, placed by `at`.
fn grid(n: u32, at: Xform) -> Shape {
    let points: Vec<[f32; 3]> = (0..=n)
        .flat_map(|j| (0..=n).map(move |i| [i as f32 / n as f32, 0.0, j as f32 / n as f32]))
        .collect();
    let row = n + 1;
    let triangles: Vec<[u32; 3]> = (0..n)
        .flat_map(|j| (0..n).map(move |i| j * row + i))
        .flat_map(|p| [[p, p + row, p + 1], [p + 1, p + row, p + row + 1]])
        .collect();
    Shape::builder(Column::from(points), Topology::Triangles(triangles.into()))
        .xform(Sampled::still(at))
        .build()
        .expect("a grid")
}

#[test]
#[ignore = "hand: a measurement"]
fn hand_run_a_moved_object_beside_many_instances() {
    let (device, queue) = engine::device();
    let mut renderer = Renderer::new(&device, &queue);
    let mut sync = Sync::default();
    // 2000 copies of a 512-triangle grid (1M triangles placed), and a box.
    let copies: Vec<Xform> = (0..2000)
        .map(|k| Xform::translation(Vec3::new((k % 50) as f64 * 1.5, 0.0, (k / 50) as f64 * 1.5)))
        .collect();
    let instancer = Instancer::builder(vec![id(0)], copies.into())
        .build()
        .expect("an Instancer");
    let update = SceneUpdate {
        added: vec![
            (id(0), Prim::Shape(grid(16, Xform::IDENTITY))),
            (id(1), Prim::Instancer(instancer)),
            (id(2), Prim::Shape(grid(4, Xform::IDENTITY))),
        ],
        ..SceneUpdate::new()
    };
    let upload = |renderer: &mut Renderer, sync: &mut Sync, update| {
        let start = Instant::now();
        let changed = sync.apply(update);
        let synced = start.elapsed();
        renderer.set_scene(sync.scene(), changed);
        queue.submit([]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("idle");
        (synced, start.elapsed())
    };
    let (synced, total) = upload(&mut renderer, &mut sync, update);
    eprintln!("first: sync {synced:?}, with the upload {total:?}");
    for step in 1..=5 {
        let at = Xform::translation(Vec3::new(0.0, f64::from(step), 0.0));
        let moved = SceneUpdate {
            changed: vec![Change::new(
                id(2),
                Dirty::TRANSFORM,
                Prim::Shape(grid(4, at)),
            )],
            ..SceneUpdate::new()
        };
        let (synced, total) = upload(&mut renderer, &mut sync, moved);
        eprintln!("moved: sync {synced:?}, with the upload {total:?}");
    }
}
