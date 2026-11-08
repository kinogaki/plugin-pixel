//! Pixel as a host drives it, on the GPU: a frame a change and none at rest,
//! one wake a frame, the grid locked to the view, the shade levels, an unlit
//! surface, a material's new color, views sharing a scene, a camera moved,
//! a deformed shape, a zero normal, final outputs, previews, the film a
//! compositor reads, and a dropped read.
use super::support::{LIMIT, at, pixel, scene, surface, unlit, view};
use kinogaki_render_api::aov::standard as aovs;
use kinogaki_render_api::display::Texture;
use kinogaki_render_api::geometry::image::{Image, metadata};
use kinogaki_render_api::geometry::{Attribute, Domain, Meaning, Vec3, names};
use kinogaki_render_api::standard::{Light, LightShape};
use kinogaki_render_api::value::Value;
use kinogaki_render_api::{
    AovDesc, AovFormat, Dirty, PreviewLighting, PreviewRequest, PreviewShape, Prim, PrimChange,
    PrimId, SceneUpdate, ViewDesc, ViewId,
};
use std::sync::Arc;

fn eye() -> Vec3 {
    Vec3::new(4.0, 3.0, 6.0)
}

#[test]
#[ignore = "gpu"]
fn a_change_renders_one_frame_and_rest_renders_none() {
    let (mut engine, _display, waiter, _device) = pixel();
    engine.update(scene());
    engine.view(ViewId(1), view(64, 48, eye(), &[]));
    waiter.until(LIMIT, || engine.latest(ViewId(1)).is_some());
    let frame = engine.latest(ViewId(1)).unwrap();
    assert_eq!(frame.size(), [64, 48]);
    assert!(frame.image.plane(0).is_some(), "its pixels, as asked");
    assert_eq!(waiter.count(), 1, "one wake for the frame");
    // The box is red where it is, the floor grey, the sky the sky's.
    let middle = at(&frame, 32, 26);
    assert!(middle[0] > middle[1] * 2.0, "the box: {middle:?}");
    // Nothing changes: no frame, no wake.
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert_eq!(waiter.count(), 1);
    // Each film pixel is a 4 × 4 block from the top left.
    for (x, y) in [(0, 0), (12, 20), (60, 44)] {
        let corner = at(&frame, x - x % 4, y - y % 4);
        assert_eq!(at(&frame, x, y), corner, "({x}, {y})");
    }
}

#[test]
#[ignore = "gpu"]
fn the_light_falls_on_the_shade_levels() {
    let (mut engine, _display, waiter, _device) = pixel();
    engine.update(scene());
    // A white sky, so the light's chromaticity is the sun's white.
    let sky = Light {
        shape: LightShape::Dome {
            image: None,
            format: kinogaki_render_api::standard::DomeFormat::Automatic,
            pole: kinogaki_render_api::standard::Pole::Scene,
            ground: None,
        },
        ..Light::default()
    };
    let white = PrimChange::new(
        PrimId {
            slot: 5,
            generation: 0,
        },
        Dirty::PARAMS,
        Prim::Declared(sky.declared()),
    );
    engine.update(SceneUpdate {
        changed: vec![white],
        ..SceneUpdate::new()
    });
    // No bounce, whose light the box would tint.
    let flat = [
        ("dither", Value::Choice("off".into())),
        ("shade_levels", Value::Integer(2)),
        ("gi", Value::Number(0.0)),
    ];
    engine.view(ViewId(1), view(64, 48, eye(), &flat));
    waiter.until(LIMIT, || engine.latest(ViewId(1)).is_some());
    let frame = engine.latest(ViewId(1)).unwrap();
    // Two levels without dither: every surface pixel's light is on a level
    // (the ramp's darkest, its brightest, or brighter in the same steps), ×
    // the floor's or the box's color; and the sky behind.
    let mut colors: Vec<[u32; 4]> = (0..48)
        .flat_map(|y| (0..64).map(move |x| (x, y)))
        .map(|(x, y)| at(&frame, x, y).map(|v| (v * 1000.0) as u32))
        .collect();
    colors.sort_unstable();
    colors.dedup();
    assert!(colors.len() <= 7, "{} colors: {colors:?}", colors.len());
}

/// MaterialX's Surface Unlit, rendered as it is: the box shows its color,
/// whatever lights it.
#[test]
#[ignore = "gpu"]
fn an_unlit_surface_is_its_color_whatever_lights_it() {
    let (mut engine, _display, waiter, _device) = pixel();
    let mut update = scene();
    update.added[1].1 = unlit([0.2, 0.6, 0.3]);
    engine.update(update);
    let flat = [("dither", Value::Choice("off".into()))];
    engine.view(ViewId(1), view(64, 48, eye(), &flat));
    waiter.until(LIMIT, || engine.latest(ViewId(1)).is_some());
    let shown = at(&engine.latest(ViewId(1)).unwrap(), 32, 26);
    let near = (shown[..3].iter().zip([0.2, 0.6, 0.3])).all(|(a, b)| (a - b).abs() < 2e-3);
    assert!(near, "{shown:?}");
}

#[test]
#[ignore = "gpu"]
fn a_new_color_renders_again_and_views_share_the_scene() {
    let (mut engine, _display, waiter, _device) = pixel();
    engine.update(scene());
    engine.view(ViewId(1), view(64, 48, eye(), &[]));
    engine.view(
        ViewId(2),
        view(
            32,
            32,
            Vec3::new(-5.0, 2.0, 2.0),
            &[("pixel_size", Value::Integer(2))],
        ),
    );
    waiter.until(LIMIT, || {
        engine.latest(ViewId(1)).is_some() && engine.latest(ViewId(2)).is_some()
    });
    let before = engine.latest(ViewId(1)).unwrap();
    let blue = SceneUpdate {
        changed: vec![PrimChange::new(
            PrimId {
                slot: 1,
                generation: 0,
            },
            Dirty::PARAMS,
            surface([0.05, 0.1, 0.7]),
        )],
        ..SceneUpdate::new()
    };
    let shown = [ViewId(1), ViewId(2)].map(|v| engine.latest(v).unwrap().generation);
    engine.update(blue);
    waiter.until(LIMIT, || {
        let views = [ViewId(1), ViewId(2)].into_iter().zip(shown);
        views
            .into_iter()
            .all(|(v, was)| engine.latest(v).is_some_and(|f| f.generation > was))
    });
    let after = engine.latest(ViewId(1)).unwrap();
    let (was, now) = (at(&before, 32, 26), at(&after, 32, 26));
    assert!(was[0] > was[2] && now[2] > now[0], "{was:?} then {now:?}");
    engine.drop_view(ViewId(2));
    assert!(engine.latest(ViewId(2)).is_none());
}

/// A camera prim moved renders again the view looking through it alone:
/// the other view, whose scene didn't change, renders no frame.
#[test]
#[ignore = "gpu"]
fn a_camera_moved_renders_the_view_through_it_alone() {
    use kinogaki_render_api::geometry::Sampled;
    use kinogaki_render_api::{Camera, Id, ViewCamera};
    let (mut engine, _display, waiter, _device) = pixel();
    let camera = |eye: Vec3| Camera {
        xform: Sampled::still(super::support::look(eye, Vec3::new(0.0, 0.75, 0.0))),
        ..Camera::default()
    };
    let lens: Id<Camera> = Id::of(PrimId {
        slot: 9,
        generation: 0,
    });
    let mut first = scene();
    first.added.push((lens.prim(), Prim::Camera(camera(eye()))));
    engine.update(first);
    let through = ViewDesc {
        camera: ViewCamera::Prim(lens),
        ..view(32, 32, eye(), &[])
    };
    engine.view(ViewId(1), through);
    engine.view(ViewId(2), view(32, 32, Vec3::new(-5.0, 2.0, 2.0), &[]));
    waiter.until(LIMIT, || {
        engine.latest(ViewId(1)).is_some() && engine.latest(ViewId(2)).is_some()
    });
    let shown = [ViewId(1), ViewId(2)].map(|v| engine.latest(v).unwrap().generation);
    engine.update(SceneUpdate {
        changed: vec![PrimChange::new(
            lens.prim(),
            Dirty::TRANSFORM,
            Prim::Camera(camera(Vec3::new(6.0, 3.0, 4.0))),
        )],
        ..SceneUpdate::new()
    });
    waiter.until(LIMIT, || {
        engine
            .latest(ViewId(1))
            .is_some_and(|f| f.generation > shown[0])
    });
    std::thread::sleep(std::time::Duration::from_millis(100));
    let other = engine.latest(ViewId(2)).unwrap().generation;
    assert_eq!(other, shown[1], "the free view renders nothing again");
}

/// The box's points moved up out of the middle and back, each a POINTS
/// change the meshes take in place: it renders where its points are.
#[test]
#[ignore = "gpu"]
fn a_deformed_shape_renders_where_its_points_moved() {
    let (mut engine, _display, waiter, _device) = pixel();
    let scene = scene();
    let (box_id, Prim::Shape(held)) = scene.added[3].clone() else {
        panic!("the box");
    };
    engine.update(scene);
    engine.view(ViewId(1), view(64, 48, eye(), &[]));
    waiter.until(LIMIT, || engine.latest(ViewId(1)).is_some());
    let mut middle = |lift: f32| {
        let points: Vec<[f32; 3]> = (held.points().iter())
            .map(|p| [p[0], p[1] + lift, p[2]])
            .collect();
        let moved = (held.rebuilt().points(points.into()).build()).expect("the box");
        let was = engine.latest(ViewId(1)).unwrap().generation;
        engine.update(SceneUpdate {
            changed: vec![PrimChange::new(box_id, Dirty::POINTS, Prim::Shape(moved))],
            ..SceneUpdate::new()
        });
        waiter.until(LIMIT, || {
            engine.latest(ViewId(1)).is_some_and(|f| f.generation > was)
        });
        at(&engine.latest(ViewId(1)).unwrap(), 32, 26)
    };
    let away = middle(20.0);
    assert!(
        away[0] < away[2] * 2.0,
        "the box gone from the middle: {away:?}"
    );
    let back = middle(0.0);
    assert!(back[0] > back[1] * 2.0, "the box back: {back:?}");
}

/// An authored zero normal shades as its triangle's own: the box holding
/// `N`s of zero renders as the box holding none (its light went NaN).
#[test]
#[ignore = "gpu"]
fn a_zero_normal_shades_as_its_triangles_own() {
    let (mut engine, _display, waiter, _device) = pixel();
    let scene = scene();
    let (box_id, Prim::Shape(held)) = scene.added[3].clone() else {
        panic!("the box");
    };
    engine.update(scene);
    engine.view(ViewId(1), view(64, 48, eye(), &[]));
    waiter.until(LIMIT, || engine.latest(ViewId(1)).is_some());
    let flat = engine.latest(ViewId(1)).unwrap();
    let zero = Attribute::tuples(vec![[0.0f32; 3]; 8], Meaning::Normal).expect("normals");
    let zeroed = held.rebuilt().primvar(Domain::Point, names::N, zero);
    engine.update(SceneUpdate {
        changed: vec![PrimChange::new(
            box_id,
            Dirty::PRIMVARS,
            Prim::Shape(zeroed.build().expect("the box")),
        )],
        ..SceneUpdate::new()
    });
    waiter.until(LIMIT, || {
        (engine.latest(ViewId(1))).is_some_and(|f| f.generation > flat.generation)
    });
    let zeroed = engine.latest(ViewId(1)).unwrap();
    assert!(
        zeroed.image.plane(0) == flat.image.plane(0),
        "{:?}",
        at(&zeroed, 32, 26)
    );
}

#[test]
#[ignore = "gpu"]
fn final_outputs_come_at_the_views_size_and_a_dropped_read_lets_go() {
    let (mut engine, _display, waiter, _device) = pixel();
    engine.update(scene());
    engine.view(ViewId(1), view(64, 48, eye(), &[]));
    let ask = |id: &str, format| AovDesc {
        aov: engine.description().aov(id).cloned().unwrap(),
        format,
    };
    let asked = [
        ask(aovs::BEAUTY, AovFormat::F32),
        ask("albedo", AovFormat::F16),
        ask(aovs::DEPTH, AovFormat::F32),
        ask("objectId", AovFormat::U32),
    ];
    let outputs = waiter.take(&engine.finish(ViewId(1), &asked), LIMIT);
    assert_eq!(
        (outputs.data().size(), outputs.layers().len()),
        ([64, 48], 4)
    );
    let names: Vec<&str> = (outputs.layers().iter())
        .map(|l| &*l.format().name)
        .collect();
    assert_eq!(names, ["", "albedo", "depth", "objectId"]);
    let ids = outputs.plane(3).expect("the ids");
    let id = |p: usize| u32::from_ne_bytes(ids[p * 4..p * 4 + 4].try_into().unwrap());
    let (box_id, sky_id) = (id(26 * 64 + 32), id(0));
    assert!(box_id > 0 && box_id < u32::MAX, "the box {box_id}");
    assert_eq!(sky_id, u32::MAX, "the sky: a miss");
    // A read no one holds any more is let go.
    drop(engine.finish(ViewId(1), &asked));
    let gone = engine.finish(ViewId(9), &asked);
    let mut outcome = None;
    waiter.until(LIMIT, || {
        outcome = gone.take();
        outcome.is_some()
    });
    assert!(
        outcome.unwrap().is_err(),
        "a view that isn't there reads nothing"
    );
}

#[test]
#[ignore = "gpu"]
fn a_preview_is_a_swatch_of_the_material() {
    let (mut engine, _display, waiter, _device) = pixel();
    let Some(kinogaki_render_api::Prim::Material(red)) = Some(surface([0.8, 0.1, 0.1])) else {
        unreachable!()
    };
    let request = |network| PreviewRequest {
        network,
        textures: Vec::new(),
        shape: PreviewShape::Sphere,
        lighting: PreviewLighting::Studio,
        side: 64,
    };
    let picture = waiter.take(&engine.preview(request(red.network.clone())), LIMIT);
    assert_eq!(picture.data().size(), [64, 64]);
    let middle = picture.read_pixel(0, 32, 32);
    assert!(middle[0] > middle[1] * 2.0, "red: {middle:?}");
    // A second request replaces the first one's scene.
    let picture = waiter.take(&engine.preview(request(Arc::default())), LIMIT);
    let middle = picture.read_pixel(0, 32, 32);
    assert!(
        middle[1] > middle[0] * 0.6,
        "the default, a grey under warm light: {middle:?}"
    );
}

#[test]
#[ignore = "gpu"]
fn a_compositor_reads_the_film_and_the_outputs_it_keeps() {
    let (mut engine, mut display, waiter, _device) = pixel();
    engine.update(scene());
    engine.view(ViewId(1), view(64, 48, eye(), &[]));
    waiter.until(LIMIT, || engine.latest(ViewId(1)).is_some());
    let frame = engine.latest(ViewId(1)).unwrap();
    let shown = display.acquire(&frame).expect("the shown image");
    assert_eq!(shown.frame().generation, frame.generation);
    drop(shown);
    let albedo = engine.description().aov("albedo").cloned().unwrap();
    let film = display
        .acquire_film(&frame, std::slice::from_ref(&albedo))
        .expect("the film");
    assert_eq!(film.data().size(), [16, 12]);
    let pixelated = (film.attributes()).get(Domain::Detail, metadata::PIXELATED);
    assert!(
        pixelated.is_some(),
        "pixel art magnifies through the nearest texel"
    );
    assert!(!holds(&film, "albedo"), "kept from a later frame on");
    drop(film);
    // Asking for it rendered the view again, keeping the albedo.
    waiter.until(LIMIT, || {
        engine
            .latest(ViewId(1))
            .is_some_and(|f| !Arc::ptr_eq(&f, &frame))
    });
    let frame = engine.latest(ViewId(1)).unwrap();
    let film = display.acquire_film(&frame, &[albedo]).expect("the film");
    assert!(holds(&film, "albedo") && film.layers()[0].format().name.is_empty());
    assert!(
        film.layers().iter().all(|l| Texture::of(l).is_some()),
        "held as textures"
    );
}

/// Whether `film` holds output `id`'s layer.
fn holds(film: &Image, id: &str) -> bool {
    (film.layers().iter()).any(|l| &*l.format().name == id)
}

/// A final read of a view whose camera went fails: never the film of the
/// camera it had, stretched to the new size.
#[test]
#[ignore = "gpu"]
fn a_final_read_through_a_camera_gone_fails() {
    let (mut engine, _display, waiter, _device) = pixel();
    engine.update(scene());
    engine.view(ViewId(1), view(64, 48, eye(), &[]));
    waiter.until(LIMIT, || engine.latest(ViewId(1)).is_some());
    let mut gone = view(32, 32, eye(), &[]);
    gone.camera = kinogaki_render_api::ViewCamera::Prim(kinogaki_render_api::Id::of(PrimId {
        slot: 99,
        generation: 0,
    }));
    engine.view(ViewId(1), gone);
    let beauty = AovDesc {
        aov: engine.description().aov(aovs::BEAUTY).cloned().unwrap(),
        format: AovFormat::F32,
    };
    let read = engine.finish(ViewId(1), &[beauty]);
    let mut outcome = None;
    waiter.until(LIMIT, || {
        outcome = read.take();
        outcome.is_some()
    });
    assert!(outcome.unwrap().is_err(), "no film of its camera");
}

/// A view dropped takes the outputs it kept with it: a later view given its
/// id keeps only what it asks for, from a later frame on.
#[test]
#[ignore = "gpu"]
fn a_dropped_views_kept_outputs_go_with_it() {
    let (mut engine, mut display, waiter, _device) = pixel();
    engine.update(scene());
    engine.view(ViewId(1), view(64, 48, eye(), &[]));
    waiter.until(LIMIT, || engine.latest(ViewId(1)).is_some());
    let albedo = engine.description().aov("albedo").cloned().unwrap();
    let first = engine.latest(ViewId(1)).unwrap();
    drop(display.acquire_film(&first, std::slice::from_ref(&albedo)));
    waiter.until(LIMIT, || {
        (engine.latest(ViewId(1)))
            .and_then(|f| display.acquire_film(&f, std::slice::from_ref(&albedo)))
            .is_some_and(|film| holds(&film, "albedo"))
    });
    engine.drop_view(ViewId(1));
    display.drop_view(ViewId(1));
    engine.view(ViewId(1), view(64, 48, eye(), &[]));
    waiter.until(LIMIT, || engine.latest(ViewId(1)).is_some());
    let frame = engine.latest(ViewId(1)).unwrap();
    let film = display.acquire_film(&frame, &[albedo]).expect("the film");
    assert!(!holds(&film, "albedo"), "nothing kept of the old view");
}
