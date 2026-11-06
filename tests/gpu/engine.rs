//! Pixel as a host drives it, on the GPU: a frame a change and none at rest,
//! one wake a frame, the grid locked to the view, the shade levels, a
//! material's new color, views sharing a scene, final outputs, previews,
//! the film a compositor reads, and a dropped read.
use super::support::{at, pixel, scene, surface, view};
use kinogaki_render_api::aov::standard as aovs;
use kinogaki_render_api::geometry::Vec3;
use kinogaki_render_api::standard::{Light, Shape as LightShape};
use kinogaki_render_api::value::Value;
use kinogaki_render_api::{
    AovDesc, AovFormat, Change, Dirty, PreviewLighting, PreviewRequest, PreviewShape, Prim, PrimId,
    SceneUpdate, ViewId,
};
use std::sync::Arc;

fn eye() -> Vec3 {
    Vec3::new(4.0, 3.0, 6.0)
}

#[test]
#[ignore = "requires a native graphics adapter; run with tools/check_all.py --gpu"]
fn a_change_renders_one_frame_and_rest_renders_none() {
    let (mut engine, _display, waiter, _device) = pixel();
    engine.update(scene());
    engine.view(ViewId(1), view(64, 48, eye(), &[]));
    waiter.until(|| engine.latest(ViewId(1)).is_some());
    let frame = engine.latest(ViewId(1)).unwrap();
    assert_eq!(
        (frame.width, frame.height, frame.pixels.len()),
        (64, 48, 64 * 48 * 4)
    );
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
#[ignore = "requires a native graphics adapter; run with tools/check_all.py --gpu"]
fn the_light_falls_on_the_shade_levels() {
    let (mut engine, _display, waiter, _device) = pixel();
    engine.update(scene());
    // A white sky, so the light's chromaticity is the sun's white.
    let sky = Light {
        shape: LightShape::Dome {
            image: None,
            ground: None,
        },
        ..Light::default()
    };
    let white = Change::new(
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
    waiter.until(|| engine.latest(ViewId(1)).is_some());
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

#[test]
#[ignore = "requires a native graphics adapter; run with tools/check_all.py --gpu"]
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
    waiter.until(|| engine.latest(ViewId(1)).is_some() && engine.latest(ViewId(2)).is_some());
    let before = engine.latest(ViewId(1)).unwrap();
    let blue = SceneUpdate {
        changed: vec![Change::new(
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
    waiter.until(|| {
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

#[test]
#[ignore = "requires a native graphics adapter; run with tools/check_all.py --gpu"]
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
    let outputs = waiter.take(&engine.finish(ViewId(1), &asked));
    assert_eq!(
        (outputs.width, outputs.height, outputs.planes.len()),
        (64, 48, 4)
    );
    for (desc, plane) in &outputs.planes {
        assert_eq!(
            plane.len(),
            64 * 48 * desc.aov.channels(),
            "{}",
            desc.aov.id
        );
    }
    let ids = &outputs.planes[3].1;
    let box_id = ids[(26 * 64 + 32) as usize].to_bits();
    let sky_id = ids[0].to_bits();
    assert!(box_id > 0 && box_id < u32::MAX, "the box {box_id}");
    assert_eq!(sky_id, u32::MAX, "the sky: a miss");
    // A read no one holds any more is let go.
    drop(engine.finish(ViewId(1), &asked));
    let gone = engine.finish(ViewId(9), &asked);
    let mut outcome = None;
    waiter.until(|| {
        outcome = gone.take();
        outcome.is_some()
    });
    assert!(
        outcome.unwrap().is_err(),
        "a view that isn't there reads nothing"
    );
}

#[test]
#[ignore = "requires a native graphics adapter; run with tools/check_all.py --gpu"]
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
    let picture = waiter.take(&engine.preview(request(red.network.clone())));
    assert_eq!((picture.side, picture.pixels.len()), (64, 64 * 64 * 4));
    let middle = &picture.pixels[(32 * 64 + 32) * 4..][..4];
    assert!(middle[0] > middle[1] * 2.0, "red: {middle:?}");
    // A second request replaces the first one's scene.
    let picture = waiter.take(&engine.preview(request(Arc::default())));
    let middle = &picture.pixels[(32 * 64 + 32) * 4..][..4];
    assert!(
        middle[1] > middle[0] * 0.6,
        "the default, a grey under warm light: {middle:?}"
    );
}

#[test]
#[ignore = "requires a native graphics adapter; run with tools/check_all.py --gpu"]
fn a_compositor_reads_the_film_and_the_outputs_it_keeps() {
    let (mut engine, mut display, waiter, _device) = pixel();
    engine.update(scene());
    engine.view(ViewId(1), view(64, 48, eye(), &[]));
    waiter.until(|| engine.latest(ViewId(1)).is_some());
    let frame = engine.latest(ViewId(1)).unwrap();
    let shown = display.acquire(&frame).expect("the shown image");
    assert_eq!(shown.frame().generation, frame.generation);
    drop(shown);
    let albedo = engine.description().aov("albedo").cloned().unwrap();
    let film = display
        .acquire_film(&frame, std::slice::from_ref(&albedo))
        .expect("the film");
    assert_eq!(film.size(), [16, 12]);
    assert!(
        film.is_pixelated(),
        "pixel art magnifies through the nearest texel"
    );
    assert!(film.image("albedo").is_none(), "kept from a later frame on");
    drop(film);
    // Asking for it rendered the view again, keeping the albedo.
    waiter.until(|| {
        engine
            .latest(ViewId(1))
            .is_some_and(|f| !Arc::ptr_eq(&f, &frame))
    });
    let frame = engine.latest(ViewId(1)).unwrap();
    let film = display.acquire_film(&frame, &[albedo]).expect("the film");
    assert!(film.image("albedo").is_some() && film.images()[0].aov.is_beauty());
}

/// A final read of a view whose camera went fails: never the film of the
/// camera it had, stretched to the new size.
#[test]
#[ignore = "requires a native graphics adapter; run with tools/check_all.py --gpu"]
fn a_final_read_through_a_camera_gone_fails() {
    let (mut engine, _display, waiter, _device) = pixel();
    engine.update(scene());
    engine.view(ViewId(1), view(64, 48, eye(), &[]));
    waiter.until(|| engine.latest(ViewId(1)).is_some());
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
    waiter.until(|| {
        outcome = read.take();
        outcome.is_some()
    });
    assert!(outcome.unwrap().is_err(), "no film of its camera");
}

/// A view dropped takes the outputs it kept with it: a later view given its
/// id keeps only what it asks for, from a later frame on.
#[test]
#[ignore = "requires a native graphics adapter; run with tools/check_all.py --gpu"]
fn a_dropped_views_kept_outputs_go_with_it() {
    let (mut engine, mut display, waiter, _device) = pixel();
    engine.update(scene());
    engine.view(ViewId(1), view(64, 48, eye(), &[]));
    waiter.until(|| engine.latest(ViewId(1)).is_some());
    let albedo = engine.description().aov("albedo").cloned().unwrap();
    let first = engine.latest(ViewId(1)).unwrap();
    drop(display.acquire_film(&first, std::slice::from_ref(&albedo)));
    waiter.until(|| {
        (engine.latest(ViewId(1)))
            .and_then(|f| display.acquire_film(&f, std::slice::from_ref(&albedo)))
            .is_some_and(|film| film.image("albedo").is_some())
    });
    engine.drop_view(ViewId(1));
    display.drop_view(ViewId(1));
    engine.view(ViewId(1), view(64, 48, eye(), &[]));
    waiter.until(|| engine.latest(ViewId(1)).is_some());
    let frame = engine.latest(ViewId(1)).unwrap();
    let film = display.acquire_film(&frame, &[albedo]).expect("the film");
    assert!(
        film.image("albedo").is_none(),
        "nothing kept of the old view"
    );
}
