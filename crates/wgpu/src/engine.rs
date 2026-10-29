//! Pixel Render as a `kinogaki_render_api::RenderEngine`: a handle that
//! queues each call for the render thread ([`crate::worker`]) and returns
//! at once, and reads the newest frames the thread publishes.

use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use kinogaki_render_api::{
    AovDesc, EngineDescription, Frame, Outputs, Pending, RenderEngine, SceneUpdate, ViewDesc,
    ViewId, Wake, pending,
};

use crate::ring::{Shared, lock};
use crate::worker::Command;

/// The engine's handle on its render thread.
pub struct PixelEngine {
    pub(crate) description: Arc<EngineDescription>,
    pub(crate) commands: Sender<Command>,
    pub(crate) shared: Arc<Mutex<Shared>>,
    pub(crate) wake: Wake,
}

impl PixelEngine {
    fn send(&self, command: Command) {
        // The thread lives as long as this handle.
        let _ = self.commands.send(command);
    }
}

impl Drop for PixelEngine {
    fn drop(&mut self) {
        self.send(Command::Stop);
    }
}

impl RenderEngine for PixelEngine {
    fn description(&self) -> &Arc<EngineDescription> {
        &self.description
    }

    fn update(&mut self, update: SceneUpdate) {
        self.send(Command::Update(update));
    }

    fn view(&mut self, view: ViewId, desc: ViewDesc) {
        self.send(Command::View(view, desc));
    }

    fn drop_view(&mut self, view: ViewId) {
        self.send(Command::Drop(view));
    }

    fn latest(&self, view: ViewId) -> Option<Arc<Frame>> {
        lock(&self.shared).frames.get(&view).cloned()
    }

    fn pause(&mut self, view: ViewId, paused: bool) {
        self.send(Command::Pause(view, paused));
    }

    fn finish(&mut self, view: ViewId, outputs: &[AovDesc]) -> Pending<Outputs> {
        let (deliver, result) = pending(self.wake.clone());
        self.send(Command::Finish(view, outputs.to_vec(), deliver));
        result
    }
}
