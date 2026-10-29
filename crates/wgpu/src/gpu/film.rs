//! Film-size companions of a view's display images: for compositing, each
//! display image keeps a copy of its frame's beauty and of the outputs a
//! host asked for, at film resolution, copied from the frame's targets at
//! its end (`copy_texture_to_texture`, no pass). They live and are pinned
//! with their display image, so a host reads them while the next frame
//! overwrites the targets.

use kinogaki_pixel_render::{Aov, Film};

use super::targets::{Target, Targets};

/// One display image's companions.
pub struct Companions {
    /// The beauty.
    pub beauty: Target,
    /// The outputs kept beside it.
    pub aovs: Vec<(Aov, Target)>,
}

/// What a companion texture is used for: written by copies, sampled by a
/// host, read back by tests.
fn usage() -> wgpu::TextureUsages {
    use wgpu::TextureUsages as U;
    U::COPY_DST | U::TEXTURE_BINDING | U::COPY_SRC
}

fn target(device: &wgpu::Device, film: Film, source: &wgpu::Texture) -> Target {
    Target::new(
        device,
        "pixel film companion",
        film.size,
        source.format(),
        usage(),
    )
}

impl Companions {
    /// The beauty's companion for `film`, keeping no outputs yet.
    pub fn new(device: &wgpu::Device, targets: &Targets) -> Companions {
        Companions {
            beauty: target(device, targets.film, &targets.beauty.texture),
            aovs: Vec::new(),
        }
    }

    /// Encode the copies of `targets`' beauty and `kept`'s outputs into
    /// these companions (making those it lacks).
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        targets: &Targets,
        kept: &[Aov],
    ) {
        for &aov in kept {
            if aov != Aov::Beauty && !self.aovs.iter().any(|(a, _)| *a == aov) {
                let source = targets.of(aov);
                self.aovs
                    .push((aov, target(device, targets.film, &source.texture)));
            }
        }
        let copy = |encoder: &mut wgpu::CommandEncoder, from: &Target, to: &Target| {
            encoder.copy_texture_to_texture(
                from.texture.as_image_copy(),
                to.texture.as_image_copy(),
                to.texture.size(),
            );
        };
        copy(encoder, &targets.beauty, &self.beauty);
        for (aov, to) in &self.aovs {
            copy(encoder, targets.of(*aov), to);
        }
    }

    /// The texture of `aov`'s companion, if kept.
    pub fn texture(&self, aov: Aov) -> Option<&wgpu::Texture> {
        if aov == Aov::Beauty {
            return Some(&self.beauty.texture);
        }
        self.aovs
            .iter()
            .find(|(a, _)| *a == aov)
            .map(|(_, t)| &t.texture)
    }

    /// Every companion: the beauty first.
    pub fn all(&self) -> impl Iterator<Item = (Aov, &wgpu::Texture)> {
        std::iter::once((Aov::Beauty, &self.beauty.texture))
            .chain(self.aovs.iter().map(|(a, t)| (*a, &t.texture)))
    }
}
