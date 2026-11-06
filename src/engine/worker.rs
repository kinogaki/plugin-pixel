//! The render thread: it applies the scene's updates, renders each view that
//! needs a frame into a free display image, publishes it once the GPU has
//! finished it, and answers final reads and previews.
use super::Message;
use super::display::{IMAGES, Ring, Shared, image_key, lock};
use super::readings::{PREVIEW, Reading, preview_view};
use crate::gpu::{Image, Renderer, Viewport, readback};
use crate::outputs::Output;
use crate::render::film::Film;
use crate::render::settings::Settings;
use crate::sync::Sync;
use kinogaki_render_api::display::wgpu;
use kinogaki_render_api::geometry::Diagnostic;
use kinogaki_render_api::{Frame, PrimId, RenderStats, SceneUpdate, ViewDesc, ViewId, Wake};
use std::collections::BTreeMap;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex};

/// One view as the render thread keeps it.
struct View {
    desc: ViewDesc,
    settings: Settings,
    viewport: Viewport,
    /// Which edit of the view the next frame shows.
    generation: u64,
    /// The edit its viewport's film shows, once one rendered: what a read
    /// answers with only while it is still the view's.
    filmed: Option<u64>,
    /// How often the host had dropped its id when it gave it.
    drops: u64,
    /// It needs a frame.
    dirty: bool,
    paused: bool,
    /// The outputs its images keep beside the beauty.
    kept: Vec<Output>,
}

/// The render thread's state.
pub(super) struct Worker {
    renderer: Renderer,
    sync: Sync,
    shared: Arc<Mutex<Shared>>,
    wake: Wake,
    read_pixels: bool,
    /// Updates not applied yet, folded into one.
    pending: Option<SceneUpdate>,
    views: BTreeMap<ViewId, View>,
    focused: Option<ViewId>,
    readings: Vec<(ViewId, Reading)>,
    /// The prims the last preview added.
    previewed: Vec<PrimId>,
    /// Display image sets made so far.
    sets: u64,
    /// What the scene can't render as sent, reported with the next frame;
    /// and what was reported last, so each is said once.
    problems: Vec<Diagnostic>,
    said: Vec<Diagnostic>,
}

impl Worker {
    /// A render thread's state on `device`.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shared: Arc<Mutex<Shared>>,
        wake: Wake,
        read_pixels: bool,
    ) -> Worker {
        Worker {
            renderer: Renderer::new(device, queue),
            sync: Sync::default(),
            shared,
            wake,
            read_pixels,
            pending: None,
            views: BTreeMap::new(),
            focused: None,
            readings: Vec::new(),
            previewed: Vec::new(),
            sets: 0,
            problems: Vec::new(),
            said: Vec::new(),
        }
    }

    /// Runs until the engine stops or is dropped.
    pub fn run(&mut self, receiver: &Receiver<Message>) {
        loop {
            let first = match self.has_work() {
                true => match receiver.try_recv() {
                    Ok(m) => Some(m),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => return,
                },
                false => match receiver.recv() {
                    Ok(m) => Some(m),
                    Err(_) => return,
                },
            };
            let mut next = first;
            while let Some(message) = next {
                if !self.take(message) {
                    return;
                }
                next = receiver.try_recv().ok();
            }
            self.apply();
            self.render();
            self.read();
        }
    }

    /// Whether something can be done without a message: a view to render
    /// into a free image, or a read whose frame is done.
    fn has_work(&self) -> bool {
        let shared = lock(&self.shared);
        let renders = (self.views.iter()).any(|(id, v)| {
            let newest = shared.frames.get(id);
            let free = (shared.rings.get(id)).is_none_or(|r| r.free(newest).is_some());
            v.dirty && !v.paused && free
        });
        renders
            || self
                .readings
                .iter()
                .any(|(id, _)| self.views.get(id).is_none_or(|v| !v.dirty))
    }

    /// Takes `message`; false to stop.
    fn take(&mut self, message: Message) -> bool {
        match message {
            Message::Update(update) => self.queue(update),
            Message::View(id, desc) => self.set_view(id, desc),
            Message::Drop(id) => {
                self.views.remove(&id);
                lock(&self.shared).rings.remove(&id);
            }
            Message::Pause(id, paused) => {
                if let Some(v) = self.views.get_mut(&id) {
                    v.paused = paused;
                }
            }
            Message::Focus(id) => self.focused = Some(id),
            Message::Finish(id, outputs, deliver) => {
                self.readings.push((id, Reading::Outputs(outputs, deliver)))
            }
            Message::Preview(request, deliver) => {
                let scene = request.scene();
                let removed = std::mem::take(&mut self.previewed);
                self.previewed = scene.update.added.iter().map(|(id, _)| *id).collect();
                let view = preview_view(&scene, request.side);
                self.queue(SceneUpdate {
                    removed,
                    ..scene.update
                });
                self.set_view(PREVIEW, view);
                self.readings.push((PREVIEW, Reading::Preview(deliver)));
            }
            Message::Keep(id, kept) => {
                if let Some(v) = self.views.get_mut(&id) {
                    v.kept = kept;
                    v.dirty = true;
                }
            }
            Message::Freed => {}
            Message::Stop => return false,
        }
        true
    }

    fn queue(&mut self, update: SceneUpdate) {
        match &mut self.pending {
            Some(pending) => pending.merge(update),
            None => self.pending = Some(update),
        }
    }

    fn set_view(&mut self, id: ViewId, desc: ViewDesc) {
        let settings = crate::settings::read(&desc.values, self.sync.up());
        match self.views.get_mut(&id) {
            Some(view) => {
                view.desc = desc;
                view.settings = settings;
                view.generation += 1;
                view.dirty = true;
            }
            None => {
                let viewport = self.renderer.viewport();
                let shared = lock(&self.shared);
                let view = View {
                    desc,
                    settings,
                    viewport,
                    generation: 0,
                    filmed: None,
                    drops: shared.drops(id),
                    dirty: true,
                    paused: false,
                    kept: shared.kept.get(&id).cloned().unwrap_or_default(),
                };
                drop(shared);
                self.views.insert(id, view);
            }
        }
    }

    /// Applies the updates waiting: every view renders again.
    fn apply(&mut self) {
        let Some(update) = self.pending.take() else {
            return;
        };
        let changed = self.sync.apply(update);
        self.renderer.set_scene(self.sync.scene(), changed);
        let problems = self.sync.problems();
        if problems != self.said {
            self.said.clone_from(&problems);
            self.problems = problems;
        }
        let up = self.sync.up();
        for view in self.views.values_mut() {
            view.settings.up = up;
            view.generation += 1;
            view.dirty = true;
        }
    }

    /// Renders each view that needs a frame, the focused one first.
    fn render(&mut self) {
        let mut order: Vec<ViewId> = self.views.keys().copied().collect();
        if let Some(at) = self
            .focused
            .and_then(|f| order.iter().position(|&id| id == f))
        {
            let focused = order.remove(at);
            order.insert(0, focused);
        }
        for id in order {
            if self.views.get(&id).is_some_and(|v| v.dirty && !v.paused) {
                self.render_view(id);
            }
        }
    }

    /// One frame of view `id`, when it has a free image.
    fn render_view(&mut self, id: ViewId) {
        let Some(view) = self.views.get_mut(&id) else {
            return;
        };
        let Some(camera) = self.sync.camera(&view.desc.camera) else {
            view.dirty = false;
            let mut shared = lock(&self.shared);
            shared.problems.insert(
                id,
                Diagnostic::error("the view's camera isn't in the scene"),
            );
            drop(shared);
            (self.wake)();
            return;
        };
        let size = [view.desc.width, view.desc.height];
        let film = Film::new(size[0], size[1], view.settings.pixel_size);
        if film.is_empty() {
            view.dirty = false;
            return;
        }
        // The image to write, in a ring sized for this film.
        let mut shared = lock(&self.shared);
        let outputs: Vec<Output> = [Output::Beauty]
            .into_iter()
            .chain(view.kept.iter().copied())
            .collect();
        let fits = shared.rings.get(&id).is_some_and(|r| {
            let image = &r.images[0];
            image.shown.texture.size().width == size[0]
                && image.shown.texture.size().height == size[1]
                && image.film[0].1.texture.size().width == film.size[0]
                && image
                    .film
                    .iter()
                    .map(|(o, _)| *o)
                    .eq(outputs.iter().copied())
        });
        if !fits {
            self.sets += 1;
            let device = self.renderer.device();
            let images = (0..IMAGES)
                .map(|_| Arc::new(Image::new(device, film, &view.kept)))
                .collect();
            shared.rings.insert(id, Ring::new(self.sets, images));
        }
        let newest = shared.frames.get(&id).cloned();
        let ring = shared.rings.get_mut(&id).expect("a ring");
        let Some(at) = ring.free(newest.as_ref()) else {
            return;
        };
        ring.holds[at] = None;
        let (image, set) = (ring.images[at].clone(), ring.set);
        drop(shared);
        let device = self.renderer.device().clone();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pixel frame"),
        });
        let rendered = self.renderer.encode(
            &mut encoder,
            &mut view.viewport,
            (&camera, size, &view.settings),
            &image,
        );
        self.renderer.queue().submit([encoder.finish()]);
        // The frame is published once the GPU has finished it.
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        view.dirty = false;
        view.filmed = rendered.is_some().then_some(view.generation);
        if rendered.is_none() || id == PREVIEW {
            let mut shared = lock(&self.shared);
            if let Some(ring) = shared.rings.get_mut(&id) {
                ring.holds[at] = None;
            }
            return;
        }
        let pixels: Arc<[u16]> = match self.read_pixels {
            true => readback::halves(&device, self.renderer.queue(), &image.shown.texture).into(),
            false => Arc::from([]),
        };
        let frame = Arc::new(Frame {
            view: id,
            generation: view.generation,
            desc: view.desc.clone(),
            width: size[0],
            height: size[1],
            pixels,
            image: image_key(set, at),
            stats: RenderStats {
                samples: 1,
                target: 1,
                problems: std::mem::take(&mut self.problems),
                ..RenderStats::default()
            },
        });
        let mut shared = lock(&self.shared);
        if shared.drops(id) != view.drops {
            // Dropped while it rendered: its frame is no one's, its problems
            // the next one's.
            self.problems.clone_from(&frame.stats.problems);
            return;
        }
        if let Some(ring) = shared.rings.get_mut(&id).filter(|r| r.set == set) {
            ring.holds[at] = Some(frame.clone());
        }
        shared.frames.insert(id, frame);
        shared.problems.remove(&id);
        drop(shared);
        (self.wake)();
    }

    /// Answers each read whose view's newest frame is done; drops those no
    /// one wants, and those of views gone.
    fn read(&mut self) {
        let readings = std::mem::take(&mut self.readings);
        for (id, reading) in readings {
            if !reading.wanted() {
                continue;
            }
            let Some(view) = self.views.get(&id) else {
                continue;
            };
            if view.dirty {
                self.readings.push((id, reading));
                continue;
            }
            // Its film only while it shows the view's newest edit: a view
            // that couldn't render it (its camera gone) has none.
            let newest = view.filmed == Some(view.generation);
            let targets = newest
                .then(|| self.renderer.targets(&view.viewport))
                .flatten();
            reading.answer(
                &self.renderer,
                targets,
                &view.desc,
                view.settings.pixel_size,
            );
        }
    }
}
