//! Pixel Render on a wgpu host's own device: the seven passes
//! ([`Renderer`], with a [`Viewport`] a view), a `kinogaki_render_api`
//! `RenderEngine` on its own render thread, and the display images its
//! frames are shown in where they are (`kinogaki_render_api::display::Display`).
//!
//! [`Pixel`] (a `kinogaki_render_api::display::Engine`) starts the engine on the
//! host's device. A frame is one pass: the engine renders a view when
//! something it shows changed (its camera, its size, the scene, the
//! settings), publishes the frame once the GPU has finished it and wakes
//! the host once. Each view keeps three display images, so the image a host
//! shows is never written; each keeps film-size copies of its frame's
//! beauty and of the outputs a compositing host asked for
//! (`Display::acquire_film`). Frames carry their pixels only when
//! `Options::read_pixels` asks (tests); final frames' outputs are read back
//! by `RenderEngine::finish`.
//!
//! ```no_run
//! # fn frame(device: &wgpu::Device, queue: &wgpu::Queue, scene: &kinogaki_pixel_render::Scene) {
//! use kinogaki_pixel_render::{Camera, Settings};
//! use kinogaki_pixel_render_wgpu::{Renderer, View, Viewport};
//! let mut renderer = Renderer::new(device, queue);
//! renderer.set_settings(Settings::default());
//! renderer.set_scene(scene); // builds the BVH
//! let mut viewport = Viewport::new(&renderer, 1);
//! let view = View { camera: Camera::default(), width: 1920, height: 1080 };
//! renderer.render(&mut viewport, &view, 0); // or `encode` into the host's encoder
//! let shown = viewport.image(0);
//! # }
//! ```

mod display;
mod engine;
mod gpu;
mod outputs;
mod ring;
mod worker;

pub use display::PixelDisplay;
pub use engine::PixelEngine;
pub use gpu::{AO_RAYS, PASSES, PassTimes, Renderer, View, Viewport, f16_to_f32, read_texture};

use std::sync::{Arc, Mutex};

use kinogaki_render_api::display::{Engine, Made, Options};
use kinogaki_render_api::{EngineDescription, Wake};

/// Pixel Render, as a wgpu host holds it: its description, and how to
/// make it on the host's device.
pub struct Pixel;

impl Engine for Pixel {
    fn description(&self) -> EngineDescription {
        kinogaki_pixel_render::description()
    }

    fn make(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        wake: Wake,
        options: &Options,
    ) -> Result<Made, String> {
        make(device, queue, wake, options)
    }
}

/// Pixel Render on `device`, waking `wake` once per result, and the
/// display its frames show in. The render thread makes its pipelines; an
/// error says why there is no thread.
fn make(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wake: Wake,
    options: &Options,
) -> Result<Made, String> {
    let (commands, receive) = std::sync::mpsc::channel();
    let shared = Arc::new(Mutex::new(ring::Shared::default()));
    let (device, queue) = (device.clone(), queue.clone());
    let thread = (shared.clone(), wake.clone(), options.read_pixels);
    std::thread::Builder::new()
        .name("pixel-render".into())
        .spawn(move || {
            let (shared, wake, read_pixels) = thread;
            let renderer = Renderer::new(&device, &queue);
            worker::Worker::new(renderer, shared, wake, read_pixels).run(receive);
        })
        .map_err(|e| format!("Pixel Render's thread did not start: {e}"))?;
    let engine = PixelEngine {
        description: Arc::new(kinogaki_pixel_render::description()),
        commands: commands.clone(),
        shared: shared.clone(),
        wake,
    };
    let display = PixelDisplay {
        shared,
        freed: commands,
    };
    Ok((Box::new(engine), Box::new(display)))
}
