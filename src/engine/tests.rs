//! Readings as the SDK's `finish` says, in whichever batch the host's calls
//! reach the render thread: a reading is its view's, so a dropped view's
//! resolves as cancelled, never with the frame of a view given its id after,
//! and one of an id with no view fails at once. The render thread is held
//! until the host has sent everything, so the calls are one batch.
use super::readings::preview_view;
use super::worker::Worker;
use super::{Own, Shared};
use crate::description::description;
use kinogaki_render_api::aov::standard as aovs;
use kinogaki_render_api::geometry::Error;
use kinogaki_render_api::geometry::image::Image;
use kinogaki_render_api::standard::{DomeFormat, Light, LightShape, Pole};
use kinogaki_render_api::thread::EngineThread;
use kinogaki_render_api::{
    AovDesc, AovFormat, Camera, Pending, Prim, PrimId, RenderEngine, SceneUpdate, ViewCamera,
    ViewDesc, ViewId, ViewMode,
};
use kinogaki_studio_api_testing::engine::{self, Waiter};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long a test waits for Pixel.
const LIMIT: Duration = Duration::from_secs(60);

/// The view the tests read.
const VIEW: ViewId = ViewId(1);

/// Pixel on this machine's GPU as `make` makes it, its render thread taking
/// nothing until `go` is sent, and its wakes.
fn held() -> (EngineThread<Own>, Sender<()>, Waiter) {
    let (device, queue) = engine::device();
    let waiter = Waiter::default();
    let shared = Arc::new(Mutex::new(Shared::default()));
    let mut worker = Worker::new(&device, &queue, shared, waiter.wake());
    let (go, gate) = channel::<()>();
    let run = move |mailbox| {
        if gate.recv().is_ok() {
            worker.run(mailbox);
        }
    };
    let start = (description(), |_| ());
    let engine = EngineThread::start("pixel", start, waiter.wake(), preview_view, run)
        .expect("Pixel's render thread");
    (engine, go, waiter)
}

/// A sky to render.
fn sky() -> SceneUpdate {
    let sky = Light {
        shape: LightShape::Dome {
            image: None,
            format: DomeFormat::Automatic,
            pole: Pole::Scene,
            ground: None,
        },
        ..Light::default()
    };
    let id = PrimId {
        slot: 0,
        generation: 0,
    };
    SceneUpdate {
        added: vec![(id, Prim::Declared(sky.declared()))],
        ..SceneUpdate::new()
    }
}

/// A final view `side` across.
fn view(side: u32) -> ViewDesc {
    ViewDesc {
        camera: ViewCamera::Free(Camera::default()),
        width: side,
        height: side,
        mode: ViewMode::Final,
        values: Arc::default(),
    }
}

/// A final read of `view`'s beauty.
fn read(engine: &mut EngineThread<Own>, view: ViewId) -> Pending<Image> {
    let beauty = AovDesc {
        aov: aovs::beauty(),
        format: AovFormat::F32,
    };
    engine.finish(view, &[beauty])
}

/// What `pending` resolves as, within [`LIMIT`].
fn outcome(waiter: &Waiter, pending: &Pending<Image>) -> Result<Image, Error> {
    let mut outcome = None;
    waiter.until(LIMIT, || {
        outcome = pending.take();
        outcome.is_some()
    });
    outcome.expect("resolved")
}

/// The view given its id after the drop renders and reads as itself.
fn reads_as_itself(engine: &mut EngineThread<Own>, waiter: &Waiter, side: u32) {
    let read = read(engine, VIEW);
    let image = outcome(waiter, &read).expect("the new view's outputs");
    assert_eq!(image.data().size(), [side, side]);
}

#[test]
#[ignore = "gpu"]
fn a_read_then_a_drop_then_a_view_of_its_id_resolves_as_cancelled() {
    let (mut engine, go, waiter) = held();
    engine.update(sky());
    engine.view(VIEW, view(8));
    let dropped = read(&mut engine, VIEW);
    engine.drop_view(VIEW);
    engine.view(VIEW, view(16));
    go.send(()).expect("the render thread");
    match outcome(&waiter, &dropped) {
        Err(why) => assert!(why.is_cancelled(), "{}", why.message()),
        Ok(image) => panic!("answered with a frame {:?}", image.data().size()),
    }
    reads_as_itself(&mut engine, &waiter, 16);
}

#[test]
#[ignore = "gpu"]
fn a_read_between_a_drop_and_a_view_of_its_id_fails_saying_no_view() {
    let (mut engine, go, waiter) = held();
    engine.update(sky());
    engine.view(VIEW, view(8));
    engine.drop_view(VIEW);
    let between = read(&mut engine, VIEW);
    engine.view(VIEW, view(16));
    go.send(()).expect("the render thread");
    match outcome(&waiter, &between) {
        Err(why) => assert_eq!(why.message(), "no view 1"),
        Ok(image) => panic!("answered with a frame {:?}", image.data().size()),
    }
    reads_as_itself(&mut engine, &waiter, 16);
}

#[test]
#[ignore = "gpu"]
fn a_read_of_an_id_never_given_fails_saying_no_view() {
    let (mut engine, go, waiter) = held();
    engine.update(sky());
    engine.view(VIEW, view(8));
    let never = read(&mut engine, ViewId(7));
    go.send(()).expect("the render thread");
    let why = outcome(&waiter, &never).expect_err("no view to read");
    assert_eq!(why.message(), "no view 7");
}
