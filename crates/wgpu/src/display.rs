//! Pixel Render's display: each frame is in one of its view's display
//! images ([`crate::ring`]) on the host's own device, shown where it is
//! and sampled nearest, so a host scaling it keeps the pixels crisp. A
//! compositing host takes the film instead: the display image's film-size
//! companions, pinned with it.

use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use kinogaki_pixel_render::Aov;
use kinogaki_render_api::Frame;
use kinogaki_render_api::display::{Display, FilmImage, Shown, ShownFilm};

use crate::ring::{Pin, Ring, Shared, image_key, image_of, lock};
use crate::worker::Command;

/// The display images a host shows frames in.
pub struct PixelDisplay {
    pub(crate) shared: Arc<Mutex<Shared>>,
    pub(crate) freed: Sender<Command>,
}

/// The ring and image index holding `frame`, if they still do.
fn holding<'a>(shared: &'a Shared, frame: &Arc<Frame>) -> Option<(&'a Ring, usize)> {
    let ring = shared.rings.get(&frame.view)?;
    let (id, at) = image_of(frame.image);
    let held = ring.holds.get(at)?.as_ref()?;
    if id != ring.id || !Arc::ptr_eq(held, frame) {
        return None;
    }
    debug_assert_eq!(image_key(id, at), frame.image);
    Some((ring, at))
}

/// A film image's key: its display image's, told apart by the output.
fn film_key(ring: &Ring, at: usize, aov: Aov) -> (u64, u32) {
    let output = Aov::ALL.iter().position(|&a| a == aov).unwrap_or(0) as u32;
    (ring.id, at as u32 + 8 * (1 + output))
}

impl Display for PixelDisplay {
    fn acquire(&mut self, frame: &Arc<Frame>) -> Option<Shown> {
        let shared = lock(&self.shared);
        let (ring, at) = holding(&shared, frame)?;
        let pin = Pin::new(ring, at, self.freed.clone());
        Some(
            Shown::new(
                frame.clone(),
                (ring.id, at as u32),
                ring.views[at].clone(),
                pin,
            )
            .filtered(wgpu::FilterMode::Nearest),
        )
    }

    fn acquire_film(
        &mut self,
        frame: &Arc<Frame>,
        aovs: &[kinogaki_render_api::Aov],
    ) -> Option<ShownFilm> {
        let wanted: Vec<Aov> = aovs
            .iter()
            .filter_map(|a| Aov::of_id(&a.id))
            .filter(|&a| a != Aov::Beauty)
            .collect();
        let mut shared = lock(&self.shared);
        // Outputs not kept yet are kept from the next frame on.
        let kept = shared.kept.entry(frame.view).or_default();
        let new: Vec<Aov> = wanted
            .iter()
            .copied()
            .filter(|a| !kept.contains(a))
            .collect();
        if !new.is_empty() {
            kept.extend(new);
            let _ = self.freed.send(Command::Keep(frame.view, kept.clone()));
        }
        let (ring, at) = holding(&shared, frame)?;
        let film = ring.films.get(at)?.as_ref()?;
        let images = film
            .images
            .iter()
            .filter(|(aov, _)| *aov == Aov::Beauty || wanted.contains(aov))
            .map(|(aov, view)| FilmImage {
                aov: aov.output(),
                view: view.clone(),
                key: film_key(ring, at, *aov),
            })
            .collect();
        let pin = Pin::new(ring, at, self.freed.clone());
        Some(ShownFilm::new(frame.clone(), film.size, images, pin))
    }
}
