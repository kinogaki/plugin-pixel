//! Pixel as a [`RenderEngine`]: a render thread that owns the renderer and
//! the scene, and never blocks its caller.
//!
//! Every call sends a message and returns. The render thread folds the
//! updates it has not applied into one ([`SceneUpdate::merge`]) and applies
//! them between frames. Pixel is realtime: a frame is one whole pass, so a
//! view renders once when something it shows changes (its desc, the scene)
//! and then rests. Views take turns, the focused one first; a paused one
//! waits. Each frame is published once the GPU has finished it, waking the
//! host once. With nothing to do the thread sleeps on its channel.
//!
//! [`RenderEngine::finish`] reads a view's outputs at the view's size once
//! the frame of its newest desc has rendered; [`RenderEngine::preview`]
//! renders a material's swatch as a view of its own, its scene replacing the
//! last request's.
mod display;
mod readings;
mod worker;

pub use display::PixelDisplay;

use crate::description::description;
use crate::outputs::Output;
use display::Shared;
use kinogaki_render_api::display::{Engine, Made, Options, wgpu};
use kinogaki_render_api::geometry::{Diagnostic, Error};
use kinogaki_render_api::{
    AovDesc, Deliver, EngineDescription, FinalOutputs, Frame, Pending, PreviewImage,
    PreviewRequest, RenderEngine, SceneUpdate, ViewDesc, ViewId, Wake, pending,
};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// What the engine's handle and display tell the render thread.
pub(crate) enum Message {
    Update(SceneUpdate),
    View(ViewId, ViewDesc),
    Drop(ViewId),
    Pause(ViewId, bool),
    Focus(ViewId),
    Finish(ViewId, Vec<AovDesc>, Deliver<FinalOutputs>),
    Preview(PreviewRequest, Deliver<PreviewImage>),
    /// Keep these outputs beside the view's display images.
    Keep(ViewId, Vec<Output>),
    /// A host let go of a display image.
    Freed,
    Stop,
}

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
        let (sender, receiver) = channel();
        let shared = Arc::new(Mutex::new(Shared::default()));
        let mut worker = worker::Worker::new(
            device,
            queue,
            shared.clone(),
            wake.clone(),
            options.read_pixels,
        );
        let thread = std::thread::Builder::new()
            .name("pixel".into())
            .spawn(move || worker.run(&receiver))
            .map_err(|e| {
                Error::invalid(format_args!("Pixel's render thread did not start: {e}"))
            })?;
        let engine = PixelEngine {
            description: Arc::new(description()),
            sender: sender.clone(),
            shared: shared.clone(),
            wake,
            worker: Some(thread),
        };
        let display = PixelDisplay {
            shared,
            freed: sender,
        };
        Ok((Box::new(engine), Box::new(display)))
    }
}

/// The engine's handle.
pub struct PixelEngine {
    description: Arc<EngineDescription>,
    sender: Sender<Message>,
    shared: Arc<Mutex<Shared>>,
    wake: Wake,
    worker: Option<JoinHandle<()>>,
}

impl PixelEngine {
    fn send(&self, message: Message) {
        // The render thread outlives every sender but the engine's drop.
        let _ = self.sender.send(message);
    }
}

impl Drop for PixelEngine {
    fn drop(&mut self) {
        self.send(Message::Stop);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl RenderEngine for PixelEngine {
    fn description(&self) -> &Arc<EngineDescription> {
        &self.description
    }

    fn update(&mut self, update: SceneUpdate) {
        self.send(Message::Update(update));
    }

    fn view(&mut self, view: ViewId, desc: ViewDesc) {
        self.send(Message::View(view, desc));
    }

    fn drop_view(&mut self, view: ViewId) {
        let mut shared = display::lock(&self.shared);
        shared.frames.remove(&view);
        shared.problems.remove(&view);
        drop(shared);
        self.send(Message::Drop(view));
    }

    fn latest(&self, view: ViewId) -> Option<Arc<Frame>> {
        display::lock(&self.shared).frames.get(&view).cloned()
    }

    fn pause(&mut self, view: ViewId, paused: bool) {
        self.send(Message::Pause(view, paused));
    }

    fn focus(&mut self, view: ViewId) {
        self.send(Message::Focus(view));
    }

    fn problem(&self, view: ViewId) -> Option<Diagnostic> {
        display::lock(&self.shared).problems.get(&view).cloned()
    }

    fn finish(&mut self, view: ViewId, outputs: &[AovDesc]) -> Pending<FinalOutputs> {
        let (deliver, result) = pending(self.wake.clone());
        self.send(Message::Finish(view, outputs.to_vec(), deliver));
        result
    }

    fn preview(&mut self, request: PreviewRequest) -> Pending<PreviewImage> {
        let (deliver, result) = pending(self.wake.clone());
        self.send(Message::Preview(request, deliver));
        result
    }
}
