//! An edit beside a large scene: a deform writes its mesh where it lies;
//! how long a moved object beside a large Instancer, and a small shape's
//! points moved beside 1.6 million triangles, take to reach the GPU (the
//! scene sync and the upload), measured by hand
//! (`cargo test --release --test gpu -- --ignored hand_run --nocapture`).
use kinogaki_pixel::gpu::Renderer;
use kinogaki_pixel::sync::Sync;
use kinogaki_render_api::display::wgpu;
use kinogaki_render_api::geometry::{Column, Sampled, Vec3, Xform};
use kinogaki_render_api::{Change, Dirty, Instancer, Prim, PrimId, SceneUpdate, Shape, Topology};
use kinogaki_studio_api_testing::engine;
use std::time::{Duration, Instant};

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
        (renderer.set_scene(sync.scene(), changed)).expect("the scene fits");
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

/// The sync and the upload of `update`, waited for on the device.
fn uploaded(renderer: &mut Renderer, sync: &mut Sync, update: SceneUpdate) -> Duration {
    let start = Instant::now();
    let changed = sync.apply(update);
    (renderer.set_scene(sync.scene(), changed)).expect("the scene fits");
    renderer.queue().submit([]);
    (renderer.device())
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("idle");
    start.elapsed()
}

/// A shape's points moved write its mesh where it lies; a shape added lays
/// the meshes out again.
#[test]
#[ignore = "gpu"]
fn a_deform_writes_its_mesh_in_place() {
    let (device, queue) = engine::device();
    let mut renderer = Renderer::new(&device, &queue);
    let mut sync = Sync::default();
    let update = SceneUpdate {
        added: vec![
            (id(0), Prim::Shape(grid(16, Xform::IDENTITY))),
            (id(1), Prim::Shape(grid(4, Xform::IDENTITY))),
        ],
        ..SceneUpdate::new()
    };
    uploaded(&mut renderer, &mut sync, update);
    let laid = renderer.layouts();
    let small = grid(4, Xform::IDENTITY);
    let points: Vec<[f32; 3]> = (small.points().iter())
        .map(|p| [p[0], p[1] + 0.5, p[2]])
        .collect();
    let moved = (small.rebuilt().points(points.into()).build()).expect("a grid");
    let update = SceneUpdate {
        changed: vec![Change::new(id(1), Dirty::POINTS, Prim::Shape(moved))],
        ..SceneUpdate::new()
    };
    uploaded(&mut renderer, &mut sync, update);
    assert_eq!(renderer.layouts(), laid, "written in place");
    let added = SceneUpdate {
        added: vec![(id(2), Prim::Shape(grid(2, Xform::IDENTITY)))],
        ..SceneUpdate::new()
    };
    uploaded(&mut renderer, &mut sync, added);
    assert_eq!(renderer.layouts(), laid + 1, "laid out again");
}

/// A 128-triangle grid's points moved beside a grid of 1.6 million
/// triangles: the moved mesh's range is written, nothing else is laid out
/// again. The median of ten steps, within 20 ms.
#[test]
#[ignore = "hand: a measurement"]
fn hand_run_a_small_deform_beside_a_large_scene() {
    let (device, queue) = engine::device();
    let mut renderer = Renderer::new(&device, &queue);
    let mut sync = Sync::default();
    let large = Xform::translation(Vec3::new(2.0, 0.0, 0.0));
    let update = SceneUpdate {
        added: vec![
            (id(0), Prim::Shape(grid(900, large))),
            (id(1), Prim::Shape(grid(8, Xform::IDENTITY))),
        ],
        ..SceneUpdate::new()
    };
    let first = uploaded(&mut renderer, &mut sync, update);
    let mut steps = Vec::new();
    for step in 1..=10 {
        let lift = step as f32 * 0.01;
        let small = grid(8, Xform::IDENTITY);
        let points: Vec<[f32; 3]> = (small.points().iter())
            .map(|p| [p[0], p[1] + lift * p[0], p[2]])
            .collect();
        let moved = (small.rebuilt())
            .points(Column::from(points))
            .build()
            .expect("a grid");
        let update = SceneUpdate {
            changed: vec![Change::new(id(1), Dirty::POINTS, Prim::Shape(moved))],
            ..SceneUpdate::new()
        };
        steps.push(uploaded(&mut renderer, &mut sync, update));
    }
    steps.sort();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let median = steps[steps.len() / 2];
    println!(
        "MEASURE pixel deform of 128 triangles beside 1.6M: median {:.1} ms, best {:.1} ms (first scene {:.0} ms)",
        ms(median),
        ms(steps[0]),
        ms(first)
    );
    assert!(median <= Duration::from_millis(20), "{median:?}");
}
