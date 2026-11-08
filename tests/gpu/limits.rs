//! Pixel at the edges of the device it shares with its host (wgpu's default
//! limits, which Studio asks for): a scene whose triangles take more than a
//! buffer holds fails its views saying why, and renders once it fits; as
//! many triangles placed as copies of one render; a film of a new height
//! gets images of that height; lit fog over no surface renders.
use super::support::{LIMIT, id, pixel, scene, view};
use kinogaki_render_api::aov::standard as aovs;
use kinogaki_render_api::geometry::{Column, Sampled, Vec3, Xform};
use kinogaki_render_api::standard::{self, Light, LightShape};
use kinogaki_render_api::value::Value;
use kinogaki_render_api::{
    AovDesc, AovFormat, Declared, Instancer, Prim, RenderEngine, SceneUpdate, Shape, Topology,
    ViewId, ViewMode,
};
use kinogaki_studio_api_testing::engine::Waiter;

const VIEW: ViewId = ViewId(1);

/// Quads a side of a grid: 100,352 triangles.
const SIDE: u32 = 224;

/// Grids in the large scene: 6,422,528 triangles.
const GRIDS: u32 = 64;

fn eye() -> Vec3 {
    Vec3::new(30.0, 20.0, 40.0)
}

/// A grid of [`SIDE`] × [`SIDE`] quads over the square -1..1 of the floor,
/// two triangles a quad.
fn grid() -> Shape {
    let points = SIDE + 1;
    let at = |i: u32| -1.0 + 2.0 * i as f32 / SIDE as f32;
    let positions: Vec<[f32; 3]> = (0..points * points)
        .map(|p| [at(p % points), 0.0, at(p / points)])
        .collect();
    let triangles: Vec<[u32; 3]> = (0..SIDE * SIDE)
        .flat_map(|q| {
            let c = q / SIDE * points + q % SIDE;
            [[c, c + points, c + 1], [c + 1, c + points, c + points + 1]]
        })
        .collect();
    Shape::builder(
        Column::from(positions),
        Topology::Triangles(triangles.into()),
    )
    .path("/Grid")
    .build()
    .expect("a grid")
}

/// Where grid `k` of [`GRIDS`] lies: in rows of eight, two units apart.
fn place(k: u32) -> Xform {
    Xform::translation(Vec3::new(
        f64::from(k % 8) * 2.0,
        0.0,
        f64::from(k / 8) * 2.0,
    ))
}

/// What a read of the view's beauty resolves as.
fn read(engine: &mut Box<dyn RenderEngine>, waiter: &Waiter) -> Result<(), String> {
    let beauty = AovDesc {
        aov: engine.description().aov(aovs::BEAUTY).cloned().unwrap(),
        format: AovFormat::F32,
    };
    let pending = engine.finish(VIEW, &[beauty]);
    let mut outcome = None;
    waiter.until(LIMIT, || {
        outcome = pending.take();
        outcome.is_some()
    });
    outcome
        .expect("taken")
        .map(drop)
        .map_err(|e| e.message().to_owned())
}

#[test]
#[ignore = "gpu"]
fn a_scene_past_the_devices_buffers_fails_its_views_saying_why() {
    let (mut engine, _display, waiter, _device) = pixel();
    let grid = grid();
    let grids = (0..GRIDS).map(|k| {
        let placed = grid.rebuilt().xform(Sampled::still(place(k))).build();
        (id(k), Prim::Shape(placed.expect("a grid")))
    });
    engine.update(SceneUpdate {
        added: grids.collect(),
        ..SceneUpdate::new()
    });
    let mut desc = view(64, 48, eye(), &[]);
    desc.mode = ViewMode::Final;
    engine.view(VIEW, desc);
    waiter.until(LIMIT, || engine.problem(VIEW).is_some());
    let why = engine.problem(VIEW).expect("a problem").message;
    assert!(
        why.starts_with("the scene is too large for Pixel on this GPU"),
        "{why}"
    );
    // A final read fails saying so, at once: the render thread lives on.
    assert_eq!(read(&mut engine, &waiter), Err(why));
    // One grid fits: the view renders again.
    engine.update(SceneUpdate {
        removed: (1..GRIDS).map(id).collect(),
        ..SceneUpdate::new()
    });
    assert_eq!(read(&mut engine, &waiter), Ok(()));
    assert_eq!(engine.problem(VIEW), None);
}

#[test]
#[ignore = "gpu"]
fn as_many_triangles_placed_as_copies_of_one_render() {
    let (mut engine, _display, waiter, _device) = pixel();
    let placed = (0..GRIDS).map(place).collect::<Vec<_>>();
    let copies = Instancer::builder(vec![id(0)], placed.into())
        .path("/Grids")
        .build()
        .expect("copies of the grid");
    engine.update(SceneUpdate {
        added: vec![
            (id(0), Prim::Shape(grid())),
            (id(1), Prim::Instancer(copies)),
        ],
        ..SceneUpdate::new()
    });
    let mut desc = view(64, 48, eye(), &[]);
    desc.mode = ViewMode::Final;
    engine.view(VIEW, desc);
    assert_eq!(read(&mut engine, &waiter), Ok(()));
    assert_eq!(engine.problem(VIEW), None);
}

#[test]
#[ignore = "gpu"]
fn a_film_of_a_new_height_gets_images_of_that_height() {
    let (mut engine, mut display, waiter, _device) = pixel();
    engine.update(scene());
    // A 44 × 36 view: a pixel size of 20 makes a 3 × 2 film, 16 a 3 × 3
    // one, the same width: taller (copied into images of the old height, it
    // ran past them), then shorter again (it read their stale rows).
    for (size, film) in [(20, [3, 2]), (16, [3, 3]), (20, [3, 2])] {
        let desc = view(
            44,
            36,
            Vec3::new(4.0, 3.0, 6.0),
            &[("pixel_size", Value::Integer(size))],
        );
        engine.view(VIEW, desc.clone());
        waiter.until(LIMIT, || {
            engine.latest(VIEW).is_some_and(|f| f.desc == desc)
        });
        let frame = engine.latest(VIEW).expect("a frame");
        let read = display.acquire_film(&frame, &[]).expect("its film");
        assert_eq!(read.data().size(), film, "at pixel size {size}");
    }
}

#[test]
#[ignore = "gpu"]
fn lit_fog_over_no_surface_renders() {
    let (mut engine, _display, waiter, _device) = pixel();
    // Its shadow and sky rays trace a scene of no instances.
    let sun = Light {
        shape: LightShape::Distant { angle: 0.0 },
        intensity: 3.0,
        ..Light::default()
    };
    let mut fog = Declared::new(standard::FOG, standard::fog().schema().values());
    fog.values.insert("density".into(), Value::Number(0.5));
    fog.values.insert("inscatter".into(), Value::Bool(true));
    engine.update(SceneUpdate {
        added: vec![
            (id(0), Prim::Declared(sun.declared())),
            (id(1), Prim::Declared(fog)),
        ],
        ..SceneUpdate::new()
    });
    let mut desc = view(32, 32, eye(), &[]);
    desc.mode = ViewMode::Final;
    engine.view(VIEW, desc);
    assert_eq!(read(&mut engine, &waiter), Ok(()));
}
