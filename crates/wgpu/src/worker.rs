//! The render thread: it takes the engine's commands, applies them at a
//! frame boundary and renders every view something changed for, then
//! publishes their frames and wakes the host once.
//!
//! It blocks on its command channel between frames (nothing polls). Scene
//! updates waiting together merge (`SceneUpdate::merge`) and apply once
//! ([`Store`] says what changed: only geometry rebuilds the BVH); a view's
//! newest desc wins. Each round's views encode into one submission, which
//! the thread waits on before it publishes: a frame is done after its one
//! pass, with its GPU time known. A view waiting for a free display image
//! renders when a `Shown` lets one go.

use std::collections::BTreeMap;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use kinogaki_pixel_render::{Aov, Settings, Store};
use kinogaki_render_api::{
    AovDesc, Deliver, Frame, Outputs, RenderStats, SceneUpdate, ViewCamera, ViewDesc, ViewId,
    ViewMode, Wake,
};

use crate::gpu::{Renderer, View, Viewport};
use crate::ring::{FilmViews, IMAGES, Ring, Shared, image_key, lock};

/// What the engine asks of its render thread.
pub enum Command {
    /// Scene edits.
    Update(SceneUpdate),
    /// A view added or changed.
    View(ViewId, ViewDesc),
    /// A view removed.
    Drop(ViewId),
    /// A view paused or resumed.
    Pause(ViewId, bool),
    /// A view's outputs, read after its next frame.
    Finish(ViewId, Vec<AovDesc>, Deliver<Outputs>),
    /// A view's display images keep film-size copies of these outputs
    /// (beside the beauty) from its next frame on.
    Keep(ViewId, Vec<Aov>),
    /// A display image was let go.
    Freed,
    /// The engine was dropped: the thread ends (pins still held only
    /// unpin).
    Stop,
}

/// One view on the render thread.
struct Slot {
    desc: ViewDesc,
    viewport: Viewport,
    /// Counts the edits of the view (its desc, the scene) it has seen.
    generation: u64,
    /// Something changed since its last frame.
    dirty: bool,
    paused: bool,
    finish: Option<(Vec<AovDesc>, Deliver<Outputs>)>,
}

/// The render thread's state.
pub struct Worker {
    pub renderer: Renderer,
    pub store: Store,
    views: BTreeMap<ViewId, Slot>,
    pub shared: Arc<Mutex<Shared>>,
    pub wake: Wake,
    pub read_pixels: bool,
}

/// A view chosen for this round: its camera and the display image it
/// writes.
struct Planned {
    view: ViewId,
    camera: kinogaki_pixel_render::Camera,
    image: usize,
}

impl Worker {
    /// A render thread's state of an empty scene.
    pub fn new(
        renderer: Renderer,
        shared: Arc<Mutex<Shared>>,
        wake: Wake,
        read_pixels: bool,
    ) -> Worker {
        Worker {
            renderer,
            store: Store::new(),
            views: BTreeMap::new(),
            shared,
            wake,
            read_pixels,
        }
    }

    /// Take commands until the engine is dropped.
    pub fn run(mut self, commands: Receiver<Command>) {
        while let Ok(first) = commands.recv() {
            let mut update = SceneUpdate::new();
            let mut stop = self.take(first, &mut update);
            while !stop && let Ok(next) = commands.try_recv() {
                stop = self.take(next, &mut update);
            }
            if stop {
                return;
            }
            if !update.is_empty() {
                self.apply(update);
            }
            self.render();
        }
    }

    /// Take `command` (scene edits merge into `update`); true to stop.
    fn take(&mut self, command: Command, update: &mut SceneUpdate) -> bool {
        match command {
            Command::Update(next) => update.merge(next),
            Command::View(view, desc) => match self.views.get_mut(&view) {
                Some(slot) => {
                    if slot.desc != desc {
                        slot.desc = desc;
                        slot.generation += 1;
                        slot.dirty = true;
                    }
                }
                None => {
                    let slot = Slot {
                        desc,
                        viewport: Viewport::new(&self.renderer, IMAGES),
                        generation: 1,
                        dirty: true,
                        paused: false,
                        finish: None,
                    };
                    self.views.insert(view, slot);
                }
            },
            Command::Drop(view) => {
                if let Some((_, deliver)) = self.views.remove(&view).and_then(|s| s.finish) {
                    deliver.deliver(Outputs::default());
                }
                let mut shared = lock(&self.shared);
                shared.frames.remove(&view);
                shared.rings.remove(&view);
                shared.kept.remove(&view);
            }
            Command::Keep(view, aovs) => {
                if let Some(slot) = self.views.get_mut(&view)
                    && slot.viewport.kept() != aovs
                {
                    slot.viewport.keep(&aovs);
                    slot.dirty = true;
                }
            }
            Command::Pause(view, paused) => {
                if let Some(slot) = self.views.get_mut(&view) {
                    slot.paused = paused;
                }
            }
            Command::Finish(view, outputs, deliver) => match self.views.get_mut(&view) {
                Some(slot) => {
                    slot.finish = Some((outputs, deliver));
                    slot.dirty = true;
                }
                None => deliver.deliver(Outputs::default()),
            },
            Command::Freed => {}
            Command::Stop => return true,
        }
        false
    }

    /// Apply merged scene edits: only what changed reaches the renderer.
    fn apply(&mut self, update: SceneUpdate) {
        let changed = self.store.apply(update);
        if !changed.any() {
            return;
        }
        if changed.materials {
            self.renderer.set_materials(&self.store.materials());
        }
        if changed.geometry {
            self.renderer.set_meshes(&self.store.meshes());
        }
        if changed.geometry || changed.materials {
            self.renderer.set_volumes(&self.store.volumes());
        }
        if changed.lights {
            self.renderer.set_lights(&self.store.lights());
        }
        if changed.fog {
            self.renderer.set_fog(self.store.fog());
        }
        if changed.settings {
            self.renderer
                .set_settings(Settings::from_render(self.store.settings()));
        }
        for slot in self.views.values_mut() {
            slot.generation += 1;
            slot.dirty = true;
        }
    }

    /// The views to render this round, each with a display image it may
    /// write (marked as being written).
    fn plan(&mut self) -> Vec<Planned> {
        let mut shared = lock(&self.shared);
        let mut planned = Vec::new();
        for (&view, slot) in &self.views {
            if !slot.dirty || (slot.paused && slot.desc.mode != ViewMode::Final) {
                continue;
            }
            let camera = match &slot.desc.camera {
                ViewCamera::Free(camera) => camera.clone(),
                // Waits for its camera to be sent.
                ViewCamera::Prim(id) => match self.store.camera(*id) {
                    Some(camera) => camera,
                    None => continue,
                },
            };
            let film = self.renderer.film(slot.desc.width, slot.desc.height);
            let image = if slot.viewport.film() == Some(film) {
                let newest = shared.frames.get(&view).cloned();
                let Some(ring) = shared.rings.get_mut(&view) else {
                    continue;
                };
                // None free: it renders when a Shown lets one go.
                let Some(at) = ring.free(newest.as_ref()) else {
                    continue;
                };
                ring.holds[at] = None;
                at
            } else {
                // A new film: new images, all free.
                0
            };
            planned.push(Planned {
                view,
                camera,
                image,
            });
        }
        planned
    }

    /// Render the planned views in one submission, then publish their
    /// frames and wake the host once.
    fn render(&mut self) {
        let planned = self.plan();
        if planned.is_empty() {
            return;
        }
        let device = self.renderer.device().clone();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pixel frames"),
        });
        let mut rendered = Vec::with_capacity(planned.len());
        for p in planned {
            let slot = self.views.get_mut(&p.view).expect("a planned view");
            let view = View {
                camera: p.camera,
                width: slot.desc.width,
                height: slot.desc.height,
            };
            slot.dirty = false;
            if let Some(film) =
                self.renderer
                    .encode(&mut encoder, &mut slot.viewport, &view, p.image)
            {
                rendered.push((p.view, p.image, film));
            }
        }
        let submitted = std::time::Instant::now();
        let index = self.renderer.queue().submit([encoder.finish()]);
        // Blocks this thread until the GPU has done the round (its times
        // and read-backs need it); the host's own submissions go on.
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        });
        // A frame's time is the round's, submission to completion: pass
        // timestamps overlap where passes run side by side (PassTimes).
        let round_ms = submitted.elapsed().as_secs_f32() * 1000.0;
        let mut deliveries = Vec::new();
        for (view, image, film) in rendered {
            let slot = self.views.get_mut(&view).expect("a rendered view");
            let viewport = &slot.viewport;
            let pixels: Arc<[u16]> = if self.read_pixels {
                viewport.read_halves(image).unwrap_or_default().into()
            } else {
                Arc::from([])
            };
            let frame = Arc::new(Frame {
                view,
                generation: slot.generation,
                desc: slot.desc.clone(),
                width: film.view[0],
                height: film.view[1],
                block: film.pixel_size,
                pixels,
                image: image_key(viewport.images_id(), image),
                stats: RenderStats {
                    samples: 1,
                    target: 1,
                    pass_ms: round_ms,
                    ..RenderStats::default()
                },
            });
            if let Some((wanted, deliver)) = slot.finish.take() {
                deliveries.push((deliver, crate::outputs::read(viewport, film, &wanted)));
            }
            let mut shared = lock(&self.shared);
            let ring = shared
                .rings
                .entry(view)
                .or_insert_with(|| Ring::new(0, Vec::new()));
            if ring.id != viewport.images_id() {
                let views = (0..viewport.images())
                    .filter_map(|at| viewport.image(at))
                    .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
                    .collect();
                *ring = Ring::new(viewport.images_id(), views);
            }
            ring.holds[image] = Some(frame.clone());
            ring.films[image] = Some(FilmViews {
                size: film.size,
                images: viewport
                    .film_images(image)
                    .into_iter()
                    .map(|(aov, t)| (aov, t.create_view(&wgpu::TextureViewDescriptor::default())))
                    .collect(),
            });
            shared.frames.insert(view, frame);
        }
        // Each delivery wakes the host itself.
        let delivered = !deliveries.is_empty();
        for (deliver, outputs) in deliveries {
            deliver.deliver(outputs);
        }
        if !delivered {
            (self.wake)();
        }
    }
}
