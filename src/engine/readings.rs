//! What a finished view is read for: a final render's outputs, or a
//! preview's picture. Both are read from the view's film at the view's size,
//! each film pixel a pixel-size block, as the display images show it.
use crate::gpu::{Renderer, Targets, readback};
use crate::outputs::Output;
use kinogaki_render_api::aov::Values;
use kinogaki_render_api::geometry::Error;
use kinogaki_render_api::thread::Reading;
use kinogaki_render_api::value::{Parameters, Value};
use kinogaki_render_api::{AovClass, PreviewScene, ViewCamera, ViewDesc, ViewMode, aov};
use std::sync::Arc;

/// Film pixels across a swatch.
const SWATCH_PIXELS: u32 = 64;

/// The view rendering a preview `scene` at `side` pixels: a swatch is
/// [`SWATCH_PIXELS`] film pixels across, at Pixel's other defaults.
pub fn preview_view(scene: &PreviewScene, side: u32) -> ViewDesc {
    let mut values = Parameters::new();
    let pixel_size = (side / SWATCH_PIXELS).max(1);
    values.insert("pixel_size".into(), Value::Integer(i64::from(pixel_size)));
    ViewDesc {
        camera: ViewCamera::Prim(scene.camera),
        width: side,
        height: side,
        mode: ViewMode::Final,
        values: Arc::new(values),
    }
}

/// `film` (`channels` values a film pixel over `size` film pixels) at view
/// size `view`, each film pixel a `pixel_size` block from the top left.
fn upscaled(
    film: &[f32],
    channels: usize,
    size: [u32; 2],
    view: [u32; 2],
    pixel_size: u32,
) -> Arc<[f32]> {
    let mut out = Vec::with_capacity(view[0] as usize * view[1] as usize * channels);
    for y in 0..view[1] {
        let row = (y / pixel_size).min(size[1] - 1) * size[0];
        for x in 0..view[0] {
            let at = (row + (x / pixel_size).min(size[0] - 1)) as usize * channels;
            out.extend_from_slice(&film[at..at + channels]);
        }
    }
    out.into()
}

/// Answers `reading` from `targets` (the view's newest frame, of `desc`);
/// fails when the view has no film.
pub fn answer(
    reading: Reading,
    renderer: &Renderer,
    targets: Option<&Targets>,
    desc: &ViewDesc,
    pixel_size: u32,
) {
    let Some(targets) = targets else {
        return reading.fail(Error::invalid("the view rendered no film"));
    };
    let view = [desc.width, desc.height];
    let planes = (reading.outputs().into_iter())
        .filter_map(|d| {
            let output = Output::of_id(&d.aov.id)?;
            let texture = &targets.of(output).texture;
            let channels = d.aov.channels();
            let film = readback::floats(renderer.device(), renderer.queue(), texture, channels);
            let plane = upscaled(&film, channels, targets.film.size, view, pixel_size);
            let values = match d.aov.class {
                AovClass::Id => Values::Whole(plane.iter().map(|v| v.to_bits()).collect()),
                _ => Values::Floats(plane),
            };
            Some((d, values))
        })
        .collect::<Vec<_>>();
    match aov::image(view, &planes, &[]) {
        Ok(image) => reading.deliver(image),
        Err(why) => reading.fail(why),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_film_upscales_in_blocks_from_the_top_left() {
        // A 2 × 2 film of one channel at pixel size 3, cropped to 5 × 4.
        let film = [1.0, 2.0, 3.0, 4.0];
        let view = upscaled(&film, 1, [2, 2], [5, 4], 3);
        assert_eq!(
            &*view,
            [
                1.0, 1.0, 1.0, 2.0, 2.0, //
                1.0, 1.0, 1.0, 2.0, 2.0, //
                1.0, 1.0, 1.0, 2.0, 2.0, //
                3.0, 3.0, 3.0, 4.0, 4.0,
            ]
        );
    }
}
