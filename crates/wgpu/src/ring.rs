//! A view's display images as a host shows them, shared by the render
//! thread and the [`crate::display::PixelDisplay`]: each image's texture
//! view, the frame it holds, its film-size companions and the `Shown`s
//! pinning it.
//!
//! The render thread writes an image only when no `Shown` pins it and it
//! does not hold the view's newest frame (which a host may be about to
//! take), so a shown frame is never overwritten; three images a view leave
//! one to write while one is shown and the newest waits. A pin let go
//! wakes the render thread, which may have a frame waiting for an image.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, MutexGuard};

use kinogaki_pixel_render::Aov;
use kinogaki_render_api::{Frame, ViewId};

use crate::worker::Command;

/// Display images a view.
pub const IMAGES: usize = 3;

/// One view's display images.
pub struct Ring {
    /// Which set of images (`Viewport::images_id`).
    pub id: u64,
    /// Each image's view.
    pub views: Vec<wgpu::TextureView>,
    /// The frame each holds (none while it is written).
    pub holds: Vec<Option<Arc<Frame>>>,
    /// Each image's film-size companions: the film's size and the outputs
    /// copied with its frame, the beauty first.
    pub films: Vec<Option<FilmViews>>,
    /// `Shown`s pinning each.
    pub pins: Arc<[AtomicU32]>,
}

impl Ring {
    /// The images of a viewport's set `id`, holding nothing yet.
    pub fn new(id: u64, views: Vec<wgpu::TextureView>) -> Ring {
        let count = views.len();
        Ring {
            id,
            views,
            holds: vec![None; count],
            films: vec![None; count],
            pins: (0..count).map(|_| AtomicU32::new(0)).collect(),
        }
    }

    /// An image the render thread may write: unpinned and not holding
    /// `newest`.
    pub fn free(&self, newest: Option<&Arc<Frame>>) -> Option<usize> {
        (0..self.views.len()).find(|&at| {
            let pinned = self.pins[at].load(Ordering::Acquire) > 0;
            let holds_newest = match (&self.holds[at], newest) {
                (Some(held), Some(newest)) => Arc::ptr_eq(held, newest),
                _ => false,
            };
            !pinned && !holds_newest
        })
    }
}

/// A display image's film-size companions as a host reads them.
#[derive(Clone)]
pub struct FilmViews {
    /// The film's size, in film pixels.
    pub size: [u32; 2],
    /// Each output's view, the beauty first.
    pub images: Vec<(Aov, wgpu::TextureView)>,
}

/// What a frame's `image` says: its set and its index in it.
pub fn image_key(id: u64, at: usize) -> u64 {
    id << 3 | at as u64
}

/// The set and index an `image` key names.
pub fn image_of(key: u64) -> (u64, usize) {
    (key >> 3, (key & 7) as usize)
}

/// The newest frames and the display images, by view.
#[derive(Default)]
pub struct Shared {
    /// Each view's newest frame.
    pub frames: BTreeMap<ViewId, Arc<Frame>>,
    /// Each view's display images.
    pub rings: BTreeMap<ViewId, Ring>,
    /// The outputs each view's companions were asked to keep.
    pub kept: BTreeMap<ViewId, Vec<Aov>>,
}

/// The shared state, locked (a poisoned lock is still the state).
pub fn lock(shared: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    shared.lock().unwrap_or_else(|e| e.into_inner())
}

/// A `Shown`'s hold on its image: let go, it unpins the image and wakes
/// the render thread.
pub struct Pin {
    pins: Arc<[AtomicU32]>,
    at: usize,
    freed: Sender<Command>,
}

impl Pin {
    /// A pin on image `at` of `ring`.
    pub fn new(ring: &Ring, at: usize, freed: Sender<Command>) -> Pin {
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
        // The render thread is gone when the engine was dropped.
        let _ = self.freed.send(Command::Freed);
    }
}
