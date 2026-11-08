//! Texture prims as the pictures surfaces sample ([`picture`]): Pixel's own
//! form of a texture, straight RGBA floats of linear ACEScg, made a row at a
//! time on the process's workers (the SDK's one budget) straight from the
//! image's tiles in the file's own samples, each texel read as the SDK
//! defines a texture's texel (spread, decoded, straight) before a large
//! image is reduced as it is read, never held whole at its own size.
use crate::render::Picture;
use kinogaki_render_api::geometry::image::TextureImage;
use kinogaki_render_api::geometry::parallel;

/// The longest side, in texels, of a picture a surface samples: a larger
/// image is reduced by the least power of two that fits it (Pixel's films
/// are coarse).
pub const MAX_PICTURE: u32 = 2048;

/// `texture`'s first layer as a picture: its texels over the data window
/// (the display window, as the host sends textures), rows from the top, a
/// color one's decoded into linear ACEScg and made straight where its
/// encoding says it is premultiplied (data as stored), a grey one grey in
/// every channel; each the mean of the texels it covers when the image is
/// reduced.
pub fn picture(texture: &TextureImage) -> Picture {
    let image = &texture.image;
    let [width, height] = image.data().size();
    let mut by = 1;
    while width.max(height).div_ceil(by) > MAX_PICTURE {
        by *= 2;
    }
    let size = [width.div_ceil(by), height.div_ceil(by)];
    let mut texels = vec![[0.0f32; 4]; size[0] as usize * size[1] as usize];
    let mut rows: Vec<&mut [[f32; 4]]> = texels.chunks_mut(size[0] as usize).collect();
    let made = parallel::chunks_mut(&mut rows, 1, |first, rows| {
        for (k, row) in rows.iter_mut().enumerate() {
            row_of(texture, by, (first + k) as u32, row);
        }
        Ok::<(), std::convert::Infallible>(())
    });
    let Ok(()) = made;
    Picture::new(size, texels)
}

/// Picture row `y` of `texture` reduced by `by` into `row`: each texel the
/// mean of the `by` × `by` texels it covers (fewer at the edges), each as
/// the texture's reader takes it ([`TextureImage::texel`]).
fn row_of(texture: &TextureImage, by: u32, y: u32, row: &mut [[f32; 4]]) {
    let data = texture.image.data();
    let ([width, height], [min_x, min_y]) = (data.size(), data.min());
    let mut counts = vec![0u32; row.len()];
    for sy in y * by..((y + 1) * by).min(height) {
        for sx in 0..width {
            let texel = texture.texel(min_x + sx as i32, min_y + sy as i32);
            let into = &mut row[(sx / by) as usize];
            *into = std::array::from_fn(|k| into[k] + texel[k]);
            counts[(sx / by) as usize] += 1;
        }
    }
    for (texel, count) in row.iter_mut().zip(counts) {
        *texel = texel.map(|v| v / count.max(1) as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinogaki_render_api::geometry::image::{Decoding, Encoding, Format, Image, Rect, Sample};

    /// A `width` × `height` texture of `format` whose pixel (x, y) holds
    /// `pixel(x, y)`, its colors linear ACEScg already.
    fn image(
        width: u32,
        height: u32,
        format: Format,
        pixel: impl Fn(u32, u32) -> Vec<f32>,
    ) -> TextureImage {
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
        let decoding = (format.encoding != Encoding::Raw).then_some(Decoding::IDENTITY);
        let image = Image::builder(window).plane(format, &bytes).build();
        TextureImage {
            image: image.unwrap(),
            decoding,
        }
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

    /// An 8-bit sRGB texture's picture holds its texels decoded (sRGB's
    /// transfer function, then its matrix) before a reduction averages them,
    /// as the decoding gives them on its own.
    #[test]
    fn a_pictures_texels_are_decoded_before_they_are_averaged() {
        use kinogaki_render_api::geometry::image::Transfer;
        let decoding = Decoding {
            transfer: Transfer::SRGB,
            matrix: [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
        };
        let rgb = Format {
            channels: Format::rgba(Sample::U8, Encoding::Raw).channels[..3].into(),
            ..Format::rgba(Sample::U8, Encoding::color("sRGB", false))
        };
        let mut texture = image(2, 1, rgb.clone(), |x, _| match x {
            0 => vec![1.0, 0.0, 0.0],
            _ => vec![0.0, 0.0, 1.0],
        });
        texture.decoding = Some(decoding);
        let made = picture(&texture);
        assert_eq!(made.texels[0], [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(made.texels[1], [0.0, 0.0, 0.5, 1.0]);
        // Reduced: the mean of light, not of codes (half a code of grey
        // would be 0.214).
        let width = MAX_PICTURE * 2;
        let mut stripes = image(width, 1, rgb, |x, _| vec![(x % 2) as f32; 3]);
        stripes.decoding = Some(Decoding {
            transfer: Transfer::SRGB,
            ..Decoding::IDENTITY
        });
        let mean = picture(&stripes).texels[0];
        assert!((mean[0] - 0.5).abs() < 1e-6, "{mean:?}");
    }
}
