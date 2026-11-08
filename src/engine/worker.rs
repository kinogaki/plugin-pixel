//! The render thread: it applies the scene's updates, renders each view that
//! needs a frame into a free display image, publishes it once the GPU has
//! finished it, and answers final reads and previews.
use super::display::{IMAGES, Ring, Shared, image_key};
use super::lock;
use super::readings::answer;
use super::{Message, Own};
use crate::gpu::{Caught, Image, Renderer, Viewport, readback};
use crate::outputs::Output;
use crate::render::film::Film;
use crate::render::settings::Settings;
use crate::sync::Sync;
use kinogaki_render_api::display::wgpu;
use kinogaki_render_api::geometry::{Diagnostic, Error};
use kinogaki_render_api::thread::{Mailbox, PREVIEW, Published, Reading};
use kinogaki_render_api::{
    EngineImage, RenderStats, SceneUpdate, ViewDesc, ViewId, ViewPass, Wake,
};
use std::collections::BTreeMap;
use std::sync::mpsc::TryRecvError;
use std::sync::{Arc, Mutex};

/// One view as the render thread keeps it.
struct View {
    desc: ViewDesc,
    settings: Settings,
    viewport: Viewport,
    /// Which edit of the view its frames show (every edit makes it `Due`).
    generation: u64,
    /// Where its frame of its newest edit stands.
    frame: Frame,
    /// How often the host had dropped its id when it gave it.
    drops: u64,
    paused: bool,
    /// The outputs its images keep beside the beauty.
    kept: Vec<Output>,
}

/// Where a view's frame of its newest edit stands: what renders it and what
/// a read of it answers.
#[derive(Clone, Debug, PartialEq)]
enum Frame {
    /// It waits for a frame: reads wait.
    Due,
    /// Its film shows it: reads take their outputs from the film.
    Shown,
    /// Its film has no pixels: nothing to render, and nothing to read.
    Empty,
    /// It couldn't render: why, which its reads fail with.
    Failed(String),
}

/// The render thread's state.
pub(super) struct Worker {
    renderer: Renderer,
    sync: Sync,
    shared: Arc<Mutex<Shared>>,
    /// What it publishes: given once it runs.
    published: Arc<Mutex<Published>>,
    wake: Wake,
    /// Frames carry their pixels.
    pub(super) read_pixels: bool,
    /// Updates not applied yet, folded into one.
    pending: Option<SceneUpdate>,
    views: BTreeMap<ViewId, View>,
    focused: Option<ViewId>,
    readings: Vec<(ViewId, Reading)>,
    /// Display image sets made so far.
    sets: u64,
    /// What the scene can't render as sent, as said last: each is said
    /// once while it stands.
    said: Vec<Diagnostic>,
    /// Why the scene doesn't fit the device: every view fails with it until
    /// a scene does.
    unfit: Option<String>,
}

impl Worker {
    /// A render thread's state on `device`.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shared: Arc<Mutex<Shared>>,
        wake: Wake,
    ) -> Worker {
        Worker {
            renderer: Renderer::new(device, queue),
            sync: Sync::default(),
            shared,
            published: Arc::default(),
            wake,
            read_pixels: false,
            pending: None,
            views: BTreeMap::new(),
            focused: None,
            readings: Vec::new(),
            sets: 0,
            said: Vec::new(),
            unfit: None,
        }
    }

    /// Runs until the engine stops or is dropped.
    pub fn run(&mut self, mailbox: Mailbox<Own>) {
        self.published = mailbox.published;
        let receiver = mailbox.receiver;
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
        let (shared, published) = (lock(&self.shared), lock(&self.published));
        let renders = (self.views.iter()).any(|(id, v)| {
            let newest = published.latest(*id);
            let free = (shared.rings.get(id)).is_none_or(|r| r.free(newest).is_some());
            v.frame == Frame::Due && !v.paused && free
        });
        renders
            || (self.readings.iter())
                .any(|(id, _)| self.views.get(id).is_none_or(|v| v.frame != Frame::Due))
    }

    /// Takes `message`; false to stop.
    fn take(&mut self, message: Message) -> bool {
        match message {
            Message::Update(update) => self.queue(update),
            Message::View(id, desc, drops) => self.set_view(id, desc, drops),
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
            Message::Read(id, reading) => self.readings.push((id, reading)),
            Message::Preview(update, desc, reading) => {
                self.queue(update);
                self.set_view(PREVIEW, desc, 0);
                self.readings.push((PREVIEW, reading));
            }
            Message::Engine(Own::Keep(id, kept)) => {
                if let Some(v) = self.views.get_mut(&id) {
                    v.kept = kept;
                    v.frame = Frame::Due;
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

    /// View `id` as `desc`, given while the host had dropped its id `drops`
    /// times.
    fn set_view(&mut self, id: ViewId, desc: ViewDesc, drops: u64) {
        let settings = crate::settings::read(&desc.values, self.sync.up());
        match self.views.get_mut(&id) {
            Some(view) => {
                view.desc = desc;
                view.settings = settings;
                view.generation += 1;
                view.frame = Frame::Due;
                view.drops = drops;
            }
            None => {
                let viewport = self.renderer.viewport();
                let shared = lock(&self.shared);
                let view = View {
                    desc,
                    settings,
                    viewport,
                    generation: 0,
                    frame: Frame::Due,
                    drops,
                    paused: false,
                    kept: shared.kept.get(&id).cloned().unwrap_or_default(),
                };
                drop(shared);
                self.views.insert(id, view);
            }
        }
    }

    /// Applies the updates waiting: a view renders again when what it shows
    /// changed (the scene, or the camera it looks through). A scene that
    /// doesn't fit the device fails every view, saying why, until one does.
    fn apply(&mut self) {
        let Some(update) = self.pending.take() else {
            return;
        };
        let cameras: BTreeMap<ViewId, _> = (self.views.iter())
            .map(|(id, view)| (*id, self.sync.camera(&view.desc.camera)))
            .collect();
        let changed = self.sync.apply(update);
        let unfit = self.renderer.set_scene(self.sync.scene(), changed).err();
        let refit = unfit != self.unfit;
        self.unfit = unfit;
        let problems = self.sync.problems();
        let new: Vec<Diagnostic> = (problems.iter())
            .filter(|p| !self.said.contains(p))
            .cloned()
            .collect();
        self.said = problems;
        if !new.is_empty() {
            let mut published = lock(&self.published);
            new.into_iter().for_each(|p| published.say(p));
            drop(published);
            (self.wake)();
        }
        let (up, sync) = (self.sync.up(), &self.sync);
        for (id, view) in &mut self.views {
            view.settings.up = up;
            if changed.any() || refit || sync.camera(&view.desc.camera) != cameras[id] {
                view.generation += 1;
                view.frame = Frame::Due;
            }
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
            if (self.views.get(&id)).is_some_and(|v| v.frame == Frame::Due && !v.paused) {
                self.render_view(id);
            }
        }
    }

    /// View `id` can't render: it says why (waking the host once), and its
    /// reads fail saying so.
    fn fail(&mut self, id: ViewId, why: String) {
        let Some(view) = self.views.get_mut(&id) else {
            return;
        };
        view.frame = Frame::Failed(why.clone());
        if lock(&self.published).fail(id, Diagnostic::error(why), view.drops) {
            (self.wake)();
        }
    }

    /// An image view `id` may write, in a ring made for `film` and the
    /// outputs it keeps: the image, its set and its place in the ring.
    fn image(&mut self, id: ViewId, film: Film) -> Option<(Arc<Image>, u64, usize)> {
        let kept = &self.views.get(&id)?.kept;
        let mut shared = lock(&self.shared);
        let outputs: Vec<Output> = [Output::Beauty]
            .into_iter()
            .chain(kept.iter().copied())
            .collect();
        let fits = (shared.rings.get(&id)).is_some_and(|r| r.film == film && r.outputs == outputs);
        if !fits {
            self.sets += 1;
            let device = self.renderer.device();
            let images = (0..IMAGES)
                .map(|_| Arc::new(Image::new(device, film, kept)))
                .collect();
            let ring = Ring::new(self.sets, (film, outputs), images);
            shared.rings.insert(id, ring);
        }
        let newest = lock(&self.published).latest(id).cloned();
        let ring = shared.rings.get_mut(&id)?;
        let at = ring.free(newest.as_ref())?;
        ring.holds[at] = None;
        Some((ring.images[at].clone(), ring.set, at))
    }

    /// One frame of view `id`, when it has a free image; why not, when it
    /// can't render: the scene doesn't fit the device, its camera isn't in
    /// the scene, its film is larger than the device's textures, or the
    /// device refused the frame.
    fn render_view(&mut self, id: ViewId) {
        let Some(view) = self.views.get(&id) else {
            return;
        };
        let size = [view.desc.width, view.desc.height];
        let film = Film::new(size[0], size[1], view.settings.pixel_size);
        let camera = match (&self.unfit, self.sync.camera(&view.desc.camera)) {
            (Some(unfit), _) => Err(unfit.clone()),
            (None, None) => Err("the view's camera isn't in the scene".to_owned()),
            (None, Some(camera)) => self.renderer.fits(&film).map(|()| camera),
        };
        let camera = match camera {
            Ok(camera) => camera,
            Err(why) => return self.fail(id, why),
        };
        if film.is_empty() {
            self.views.get_mut(&id).expect("a view").frame = Frame::Empty;
            return;
        }
        let device = self.renderer.device().clone();
        let caught = Caught::on(&device);
        let Some((image, set, at)) = self.image(id, film) else {
            return;
        };
        let view = self.views.get_mut(&id).expect("a view");
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
        let polled = device.poll(wgpu::PollType::wait_indefinitely());
        let lost = polled
            .map(drop)
            .map_err(|e| format!("the GPU was lost: {e}"));
        view.frame = match rendered {
            Some(_) => Frame::Shown,
            None => Frame::Empty,
        };
        let refused = caught.end().and(lost).err();
        // A preview's frame is read, not shown.
        if view.frame != Frame::Shown || id == PREVIEW || refused.is_some() {
            let mut shared = lock(&self.shared);
            if let Some(ring) = shared.rings.get_mut(&id) {
                ring.holds[at] = None;
            }
            drop(shared);
            if let Some(why) = refused {
                self.fail(id, why);
            }
            return;
        }
        let image = match self.read_pixels {
            true => ViewPass::pixels(
                size,
                &readback::halves(&device, self.renderer.queue(), &image.shown.texture),
            ),
            false => EngineImage(image_key(set, at)).image(size),
        };
        let Ok(image) = image else {
            return;
        };
        let frame = Arc::new(ViewPass {
            view: id,
            generation: view.generation,
            desc: view.desc.clone(),
            image,
            stats: RenderStats {
                samples: 1,
                target: 1,
                ..RenderStats::default()
            },
        });
        let mut shared = lock(&self.shared);
        if let Some(ring) = shared.rings.get_mut(&id).filter(|r| r.set == set) {
            ring.holds[at] = Some(frame.clone());
        }
        drop(shared);
        if lock(&self.published).publish(frame, view.drops) {
            (self.wake)();
        }
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
            let targets = match &view.frame {
                Frame::Due => {
                    self.readings.push((id, reading));
                    continue;
                }
                Frame::Failed(why) => {
                    reading.fail(Error::invalid(why.clone()));
                    continue;
                }
                Frame::Shown => self.renderer.targets(&view.viewport),
                Frame::Empty => None,
            };
            answer(
                reading,
                &self.renderer,
                targets,
                &view.desc,
                view.settings.pixel_size,
            );
        }
    }
}
