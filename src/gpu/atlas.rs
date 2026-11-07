//! The pictures surfaces sample, packed side by side into one texture the
//! G-buffer reads texels from (half floats, rgba): shelves of pictures,
//! tallest first, at most [`SIDE`] texels a side. Pictures that don't fit
//! are halved, the largest first, until they do.
use crate::render::Picture;
use std::sync::Arc;

/// The longest side of the atlas, in texels.
pub const SIDE: u32 = 8192;

/// The pictures packed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Atlas {
    /// Its width and height (at least 1 × 1).
    pub size: [u32; 2],
    /// Its texels, rows from the top, each rgba as half floats.
    pub texels: Vec<[u16; 4]>,
    /// Each picture's corner, its size there and its mean.
    pub placed: Vec<([u32; 2], [u32; 2], [f32; 4])>,
}

impl Atlas {
    /// `pictures` packed.
    pub fn pack(pictures: &[Arc<Picture>]) -> Atlas {
        Atlas::within(pictures, SIDE)
    }

    /// `pictures` packed in at most `side` × `side` texels.
    fn within(pictures: &[Arc<Picture>], side: u32) -> Atlas {
        let mut pictures: Vec<Arc<Picture>> = pictures.to_vec();
        loop {
            if let Some(atlas) = Atlas::shelved(&pictures, side) {
                return atlas;
            }
            let largest = (0..pictures.len())
                .max_by_key(|&k| pictures[k].size[0] as u64 * pictures[k].size[1] as u64)
                .expect("a picture to halve");
            pictures[largest] = Arc::new(pictures[largest].halved());
        }
    }

    /// `pictures` on shelves, tallest first, at most `side` texels a side;
    /// none when they don't fit.
    fn shelved(pictures: &[Arc<Picture>], side: u32) -> Option<Atlas> {
        let widest = pictures.iter().map(|p| p.size[0]).max().unwrap_or(1);
        let width = widest.max(side.min(2048)).min(side);
        let mut order: Vec<usize> = (0..pictures.len()).collect();
        order.sort_by_key(|&k| std::cmp::Reverse(pictures[k].size[1]));
        let mut corners = vec![[0u32; 2]; pictures.len()];
        let (mut x, mut y, mut shelf) = (0u32, 0u32, 0u32);
        for &k in &order {
            let [w, h] = pictures[k].size;
            if w > width {
                return None;
            }
            if x + w > width {
                (x, y, shelf) = (0, y + shelf, 0);
            }
            corners[k] = [x, y];
            x += w;
            shelf = shelf.max(h);
        }
        let height = (y + shelf).max(1);
        if height > side {
            return None;
        }
        let mut texels = vec![[0u16; 4]; width as usize * height as usize];
        let mut placed = Vec::with_capacity(pictures.len());
        for (k, picture) in pictures.iter().enumerate() {
            let [cx, cy] = corners[k];
            let [w, h] = picture.size;
            for row in 0..h {
                let from = &picture.texels[(row * w) as usize..((row + 1) * w) as usize];
                let at = ((cy + row) * width + cx) as usize;
                for (to, t) in texels[at..at + w as usize].iter_mut().zip(from) {
                    *to = t.map(kinogaki_render_api::f32_to_half);
                }
            }
            placed.push((corners[k], picture.size, picture.mean));
        }
        Some(Atlas {
            size: [width, height],
            texels,
            placed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinogaki_render_api::half_to_f32;

    /// The texel of `atlas` at `(x, y)`, as floats.
    fn texel(atlas: &Atlas, x: u32, y: u32) -> [f32; 4] {
        atlas.texels[(y * atlas.size[0] + x) as usize].map(half_to_f32)
    }

    fn picture(w: u32, h: u32, value: f32) -> Arc<Picture> {
        Arc::new(Picture::new(
            [w, h],
            vec![[value, 0.5, 0.25, 1.0]; (w * h) as usize],
        ))
    }

    #[test]
    fn pictures_lie_on_shelves_and_keep_their_texels() {
        let atlas = Atlas::pack(&[picture(4, 2, 0.75), picture(3, 5, 0.125)]);
        let [(a, a_size, a_mean), (b, b_size, _)] = [atlas.placed[0], atlas.placed[1]];
        assert_eq!((a_size, b_size), ([4, 2], [3, 5]));
        // The taller first, the other beside it.
        assert_eq!((b, a), ([0, 0], [3, 0]));
        assert_eq!(atlas.size, [2048, 5]);
        assert_eq!(texel(&atlas, a[0] + 3, a[1] + 1), [0.75, 0.5, 0.25, 1.0]);
        assert_eq!(texel(&atlas, b[0], b[1] + 4)[0], 0.125);
        assert_eq!(a_mean, [0.75, 0.5, 0.25, 1.0]);
        assert_eq!(Atlas::pack(&[]).size, [2048, 1]);
    }

    #[test]
    fn what_doesnt_fit_is_halved_largest_first() {
        let big = || picture(32, 32, 1.0);
        let atlas = Atlas::within(&[big(), big(), picture(8, 8, 0.0)], 32);
        let sizes: Vec<[u32; 2]> = atlas.placed.iter().map(|p| p.1).collect();
        assert!(
            atlas.size[0] <= 32 && atlas.size[1] <= 32,
            "{:?}",
            atlas.size
        );
        assert_eq!(sizes[2], [8, 8], "the small one kept whole");
        assert!(sizes[0][0] < 32 && sizes[1][0] < 32, "{sizes:?}");
    }
}
