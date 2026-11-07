//! Pixel's display: each frame is in one of its view's display images on the
//! host's own device, shown where it is. A compositing host takes the film
//! instead: the display image's film-size copies of the frame's outputs as
//! one Image, pinned with it, its pixels meant to be seen as pixels.
//!
//! The render thread writes an image only when no host pins it and it does
//! not hold the view's newest frame (which a host may be about to take), so a
//! shown frame is never overwritten; three images a view leave one to write
//! while one is shown and the newest waits. A pin let go wakes the render
//! thread, which may have a frame waiting for an image.
use super::{Message, Own};
use crate::gpu::Image;
use crate::outputs::Output;
use crate::render::film::Film;
use kinogaki_render_api::display::{self, Display, Shown};
use kinogaki_render_api::geometry::image::{self, metadata};
use kinogaki_render_api::geometry::{Attribute, Domain, Meaning};
use kinogaki_render_api::thread::lock;
use kinogaki_render_api::{Aov, Frame, ViewId};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

/// Display images a view.
pub const IMAGES: usize = 3;

/// One view's display images.
pub struct Ring {
    /// Which set of images: new each time the film or its outputs change.
    pub set: u64,
    /// The film they are sized for.
    pub film: Film,
    /// The outputs each keeps, the beauty first: their formats.
    pub outputs: Vec<Output>,
    /// The images.
    pub images: Vec<Arc<Image>>,
    /// The frame each holds (none while it is written).
    pub holds: Vec<Option<Arc<Frame>>>,
    /// Pins on each.
    pub pins: Arc<[AtomicU32]>,
}

impl Ring {
    /// The images of set `set`, for `film` keeping `outputs`, holding
    /// nothing yet.
    pub fn new(set: u64, (film, outputs): (Film, Vec<Output>), images: Vec<Arc<Image>>) -> Ring {
        Ring {
            set,
            film,
            outputs,
            holds: vec![None; images.len()],
            pins: (0..images.len()).map(|_| AtomicU32::new(0)).collect(),
            images,
        }
    }

    /// An image the render thread may write: unpinned, not holding `newest`.
    pub fn free(&self, newest: Option<&Arc<Frame>>) -> Option<usize> {
        (0..self.images.len()).find(|&at| {
            let pinned = self.pins[at].load(Ordering::Acquire) > 0;
            let newest =
                matches!((&self.holds[at], newest), (Some(h), Some(n)) if Arc::ptr_eq(h, n));
            !pinned && !newest
        })
    }
}

/// What a frame's `image` says: its set and its index in it.
pub fn image_key(set: u64, at: usize) -> u64 {
    set << 3 | at as u64
}

/// What the display and the render thread share besides what the thread
/// publishes.
#[derive(Default)]
pub struct Shared {
    /// Each view's display images.
    pub rings: BTreeMap<ViewId, Ring>,
    /// The outputs each view's images keep beside the beauty.
    pub kept: BTreeMap<ViewId, Vec<Output>>,
}

/// A host's hold on an image: let go, it unpins it and wakes the render
/// thread.
struct Pin {
    pins: Arc<[AtomicU32]>,
    at: usize,
    freed: Sender<Message>,
}

impl Pin {
    fn new(ring: &Ring, at: usize, freed: Sender<Message>) -> Pin {
        ring.pins[at].fetch_add(1, Ordering::AcqRel);
        Pin {
            pins: ring.pins.clone(),
            at,
            freed,
        }
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        self.pins[self.at].fetch_sub(1, Ordering::AcqRel);
        // The render thread is gone once the engine is.
        let _ = self.freed.send(Message::Freed);
    }
}

/// The display images a host shows frames in.
pub struct PixelDisplay {
    pub(super) shared: Arc<Mutex<Shared>>,
    pub(super) freed: Sender<Message>,
}

/// `film` saying its pixels are meant to be seen as pixels.
fn pixelated(film: image::Image) -> Option<image::Image> {
    let one = Attribute::tuples(vec![1i32], Meaning::None).ok()?;
    let said = film
        .attributes()
        .with(Domain::Detail, metadata::PIXELATED, one);
    film.with_attributes(said.ok()?).ok()
}

/// The ring and index of the image holding `frame`, while it does.
fn holding<'a>(shared: &'a Shared, frame: &Arc<Frame>) -> Option<(&'a Ring, usize)> {
    let ring = shared.rings.get(&frame.view)?;
    let at = (frame.image & 7) as usize;
    let held = ring.holds.get(at)?.as_ref()?;
    (frame.image >> 3 == ring.set && Arc::ptr_eq(held, frame)).then_some((ring, at))
}

impl Display for PixelDisplay {
    fn acquire(&mut self, frame: &Arc<Frame>) -> Option<Shown> {
        let shared = lock(&self.shared);
        let (ring, at) = holding(&shared, frame)?;
        let view = ring.images[at].shown.view.clone();
        let pin = Pin::new(ring, at, self.freed.clone());
        Some(Shown::new(frame.clone(), (ring.set, at as u32), view, pin))
    }

    fn acquire_film(&mut self, frame: &Arc<Frame>, aovs: &[Aov]) -> Option<image::Image> {
        let wanted: Vec<Output> = (aovs.iter())
            .filter_map(|a| Output::of_id(&a.id))
            .filter(|&o| o != Output::Beauty)
            .collect();
        let mut shared = lock(&self.shared);
        // Outputs not kept yet are kept from a later frame on.
        let kept = shared.kept.entry(frame.view).or_default();
        let new: Vec<Output> = wanted
            .iter()
            .copied()
            .filter(|o| !kept.contains(o))
            .collect();
        if !new.is_empty() {
            kept.extend(new);
            let _ = (self.freed).send(Message::Engine(Own::Keep(frame.view, kept.clone())));
        }
        let (ring, at) = holding(&shared, frame)?;
        let image = &ring.images[at];
        let id = |k: usize| ring.set << 16 ^ (at + IMAGES * (k + 1)) as u64;
        let images = (image.film.iter().enumerate())
            .filter(|(_, (o, _))| *o == Output::Beauty || wanted.contains(o))
            .map(|(k, (o, target))| (o.aov(), target.view.clone(), id(k)))
            .collect();
        let pin = Pin::new(ring, at, self.freed.clone());
        let film = display::film(frame, ring.film.size, images, pin)?;
        pixelated(film)
    }

    /// The outputs the view was asked to keep go: a later view given its id
    /// keeps only what it asks for.
    fn drop_view(&mut self, view: ViewId) {
        lock(&self.shared).kept.remove(&view);
    }
}
