//! A final frame's outputs: each asked-for output read back from the film
//! and repeated over its film pixels' blocks, at the view's size (what the
//! view showed, and what an EXR of the settings' resolution holds).

use std::sync::Arc;

use kinogaki_pixel_render::{Aov, AovPixels, Film};
use kinogaki_render_api::{AovDesc, Outputs};

use crate::gpu::Viewport;

/// `wanted`'s outputs of `viewport`'s last frame of `film`; outputs Pixel
/// Render does not write are left out.
pub fn read(viewport: &Viewport, film: Film, wanted: &[AovDesc]) -> Outputs {
    let [width, height] = film.view;
    let mut planes = Vec::with_capacity(wanted.len());
    for desc in wanted {
        let Some(aov) = Aov::of_id(&desc.aov.id) else {
            continue;
        };
        let Some(image) = viewport.read_aov(aov) else {
            continue;
        };
        let keep = desc.aov.channels();
        let stride = image.channels as usize;
        let value = |texel: usize, k: usize| -> f32 {
            match &image.pixels {
                AovPixels::F32(v) => v[texel * stride + k],
                AovPixels::U32(v) => f32::from_bits(v[texel * stride + k]),
            }
        };
        let mut values = Vec::with_capacity(width as usize * height as usize * keep);
        for y in 0..height {
            for x in 0..width {
                let [i, j] = film.film_pixel(x, y);
                let texel = (j * image.width + i) as usize;
                values.extend((0..keep).map(|k| value(texel, k)));
            }
        }
        planes.push((desc.clone(), Arc::from(values)));
    }
    Outputs {
        width,
        height,
        block: film.pixel_size,
        planes,
        names: Vec::new(),
    }
}
