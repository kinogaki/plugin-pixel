//! Texture prims as the pictures surfaces sample ([`picture`]): made a row
//! at a time on the process's workers (the SDK's one budget) straight from
//! the image's tiles, a large image reduced as it is read, never held whole
//! at its own size.
use crate::render::Picture;
use kinogaki_render_api::geometry::image::{Format, Image, TILE};
use kinogaki_render_api::geometry::parallel;

/// The longest side, in texels, of a picture a surface samples: a larger
/// image is reduced by the least power of two that fits it (Pixel's films
/// are coarse).
pub const MAX_PICTURE: u32 = 2048;

/// `image`'s main layer as a picture: its texels over the data window (the
/// display window, as the host sends textures), rows from the top, made
/// straight where its encoding says the host premultiplied them (colors;
/// data comes straight), a grey one grey in every channel; each the mean of
/// the texels it covers when the image is reduced.
pub fn picture(image: &Image) -> Picture {
    let [width, height] = image.data().size();
    let mut by = 1;
    while width.max(height).div_ceil(by) > MAX_PICTURE {
        by *= 2;
    }
    let size = [width.div_ceil(by), height.div_ceil(by)];
    let mut texels = vec![[0.0f32; 4]; size[0] as usize * size[1] as usize];
    if let Some(format) = image.layers().first().map(|l| l.format()) {
        let mut rows: Vec<&mut [[f32; 4]]> = texels.chunks_mut(size[0] as usize).collect();
        let made = parallel::chunks_mut(&mut rows, 1, |first, rows| {
            for (k, row) in rows.iter_mut().enumerate() {
                row_of(image, format, by, (first + k) as u32, row);
            }
            Ok::<(), std::convert::Infallible>(())
        });
        let Ok(()) = made;
    }
    Picture::new(size, texels)
}

/// Picture row `y` of `image` (its first layer of `format`) reduced by `by`
/// into `row`: each texel the mean of the `by` × `by` image texels it covers
/// (fewer at the edges), a tile holding nothing zeros.
fn row_of(image: &Image, format: &Format, by: u32, y: u32, row: &mut [[f32; 4]]) {
    let [width, height] = image.data().size();
    let (side, pixel) = (TILE as usize, format.pixel_bytes());
    let mut counts = vec![0u32; row.len()];
    for sy in y * by..((y + 1) * by).min(height) {
        let within = (sy % TILE) as usize * side;
        for tx in 0..width.div_ceil(TILE) {
            let tile = image.tile(0, [tx, sy / TILE]);
            let across = side.min((width - tx * TILE) as usize);
            for x in 0..across {
                let sx = tx as usize * side + x;
                let texel = match tile {
                    Some(tile) => texel(format, &tile.bytes()[(within + x) * pixel..]),
                    None => [0.0; 4],
                };
                let into = &mut row[sx / by as usize];
                *into = std::array::from_fn(|k| into[k] + texel[k]);
                counts[sx / by as usize] += 1;
            }
        }
    }
    for (texel, count) in row.iter_mut().zip(counts) {
        *texel = texel.map(|v| v / count.max(1) as f32);
    }
}

/// The pixel at the start of `bytes` (of `format`) as a straight RGBA
/// texel: one channel grey, two grey and alpha, three opaque.
fn texel(format: &Format, bytes: &[u8]) -> [f32; 4] {
    let size = format.sample.bytes();
    let c = |k: usize| format.sample.read(&bytes[k * size..]);
    let rgba = match format.channels.len() {
        1 => [c(0), c(0), c(0), 1.0],
        2 => [c(0), c(0), c(0), c(1)],
        3 => [c(0), c(1), c(2), 1.0],
        _ => [c(0), c(1), c(2), c(3)],
    };
    match format.encoding.premultiplied() {
        true => straight(rgba),
        false => rgba,
    }
}

/// `rgba` unpremultiplied where its alpha is between 0 and 1.
fn straight([r, g, b, a]: [f32; 4]) -> [f32; 4] {
    match a > 0.0 && a < 1.0 {
        true => [r / a, g / a, b / a, a],
        false => [r, g, b, a],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinogaki_render_api::geometry::image::{Encoding, Rect, Sample};

    /// A `width` × `height` image of `format` whose pixel (x, y) holds
    /// `pixel(x, y)`.
    fn image(
        width: u32,
        height: u32,
        format: Format,
        pixel: impl Fn(u32, u32) -> Vec<f32>,
    ) -> Image {
        let mut bytes = Vec::new();
        for y in 0..height {
            for x in 0..width {
                for v in pixel(x, y) {
                    let mut value = vec![0; format.sample.bytes()];
                    format.sample.write(v, &mut value);
                    bytes.extend(value);
                }
            }
        }
        let window = Rect::new([0, 0], [width as i32, height as i32]).unwrap();
        Image::builder(window)
            .plane(format, &bytes)
            .build()
            .unwrap()
    }

    /// Past a tile's side, so the tiles in a row and rows of tiles land
    /// where their texels are; premultiplied colors come straight.
    #[test]
    fn a_picture_holds_the_images_texels_straight() {
        let format = Format::rgba(Sample::F16, Encoding::color("ACEScg", true));
        let made = picture(&image(70, 66, format, |x, y| {
            vec![x as f32 * 0.5, y as f32 * 0.5, 0.25, 0.5]
        }));
        assert_eq!(made.size, [70, 66]);
        for (x, y) in [(0, 0), (63, 1), (64, 64), (69, 65)] {
            let at = (y * 70 + x) as usize;
            assert_eq!(made.texels[at], [x as f32, y as f32, 0.5, 0.5]);
        }
        let grey = Format {
            channels: Format::rgba(Sample::U8, Encoding::Raw).channels[..1].into(),
            ..Format::rgba(Sample::U8, Encoding::Raw)
        };
        let one = picture(&image(1, 1, grey, |_, _| vec![1.0]));
        assert_eq!(one.texels, [[1.0; 4]], "grey, opaque");
    }

    /// A side past [`MAX_PICTURE`] is reduced by the least power of two
    /// fitting it, each texel the mean of those it covers.
    #[test]
    fn a_large_image_is_reduced_as_it_is_read() {
        let format = Format::rgba(Sample::F32, Encoding::Raw);
        let width = MAX_PICTURE * 4 - 2;
        let made = picture(&image(width, 4, format, |x, y| {
            vec![(x % 4) as f32, y as f32, 1.0, 1.0]
        }));
        assert_eq!(made.size, [MAX_PICTURE, 1], "reduced by 4");
        // Columns 0..4 of rows 0..4: x % 4 is 0 to 3, y 0 to 3.
        assert_eq!(made.texels[0], [1.5, 1.5, 1.0, 1.0]);
        // The last covers the image's last two columns alone.
        assert_eq!(made.texels[MAX_PICTURE as usize - 1], [0.5, 1.5, 1.0, 1.0]);
    }
}
