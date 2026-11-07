//! Pixel as a render engine: the SDK's [`EngineThread`] on a render thread
//! that owns the renderer and the scene, and never blocks its caller.
//!
//! The render thread folds the
//! updates it has not applied into one ([`SceneUpdate::merge`]) and applies
//! them between frames. Pixel is realtime: a frame is one whole pass, so a
//! view renders once when something it shows changes (its desc, the scene)
//! and then rests. Views take turns, the focused one first; a paused one
//! waits. Each frame is published once the GPU has finished it, waking the
//! host once. With nothing to do the thread sleeps on its channel.
//!
//! A read (a final render's outputs, a swatch's picture) takes a view's
//! outputs at the view's size once the frame of its newest desc has rendered.
//!
//! [`SceneUpdate::merge`]: kinogaki_render_api::SceneUpdate::merge
mod display;
mod readings;
mod worker;

pub use display::PixelDisplay;

use crate::description::description;
use crate::outputs::Output;
use display::Shared;
use kinogaki_render_api::display::{Engine, Made, Options, wgpu};
use kinogaki_render_api::geometry::Error;
use kinogaki_render_api::thread::EngineThread;
use kinogaki_render_api::{EngineDescription, ViewId, Wake};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// What Pixel's display tells its render thread besides a host's calls.
pub(crate) enum Own {
    /// Keep these outputs beside the view's display images.
    Keep(ViewId, Vec<Output>),
}

/// What the render thread is sent.
pub(crate) type Message = kinogaki_render_api::thread::Message<Own>;

/// Pixel, as a wgpu host makes it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pixel;

impl Engine for Pixel {
    fn description(&self) -> EngineDescription {
        description()
    }

    /// Pixel on `device`, waking `wake` once per result, and the display its
    /// frames show in.
    fn make(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        wake: Wake,
        options: &Options,
    ) -> Result<Made, Error> {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let mut worker = worker::Worker::new(device, queue, shared.clone(), wake.clone());
        worker.read_pixels = options.read_pixels;
        let run = move |mailbox| worker.run(mailbox);
        let engine = EngineThread::start(
            "pixel",
            (description(), |_| ()),
            wake,
            readings::preview_view,
            run,
        )
        .map_err(|e| Error::invalid(format_args!("Pixel's render thread did not start: {e}")))?;
        let display = PixelDisplay {
            shared,
            freed: engine.sender(),
        };
        Ok((Box::new(engine), Box::new(display)))
    }
}

/// `mutex` locked, whatever a panic left in it.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
