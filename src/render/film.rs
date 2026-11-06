//! The film: the low-resolution image Pixel Render renders, and how it maps
//! onto the view.
//!
//! The film is `ceil(view / pixel_size)` pixels. Film pixel (i, j) covers
//! view pixels `[i·s, (i+1)·s) × [j·s, (j+1)·s)`, anchored at the view's
//! top-left corner, so the grid stays locked to the pane: the last column
//! and row may run past the view's right and bottom edges, and are cropped
//! there rather than stretched.

/// A film for a view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Film {
    /// The view's size, in view pixels.
    pub view: [u32; 2],
    /// View pixels a film pixel covers, along each axis.
    pub pixel_size: u32,
    /// The film's size, in film pixels.
    pub size: [u32; 2],
}

impl Film {
    /// The film of a `width` × `height` view at `pixel_size` (taken as at
    /// least 1).
    pub fn new(width: u32, height: u32, pixel_size: u32) -> Film {
        let s = pixel_size.max(1);
        Film {
            view: [width, height],
            pixel_size: s,
            size: [width.div_ceil(s), height.div_ceil(s)],
        }
    }

    /// Whether either side is zero (nothing to render).
    pub fn is_empty(&self) -> bool {
        self.size[0] == 0 || self.size[1] == 0
    }

    /// The film pixel showing view pixel (`x`, `y`): the upscale's nearest
    /// mapping, exact in integers.
    pub fn film_pixel(&self, x: u32, y: u32) -> [u32; 2] {
        [x / self.pixel_size, y / self.pixel_size]
    }

    /// View pixels the film covers along each axis (at least the view's
    /// size, less than one film pixel more).
    pub fn covered(&self) -> [u32; 2] {
        self.size.map(|n| n * self.pixel_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn film_size_is_the_ceiling() {
        assert_eq!(Film::new(640, 360, 4).size, [160, 90]);
        assert_eq!(Film::new(641, 361, 4).size, [161, 91]);
        assert_eq!(Film::new(1920, 1080, 4).size, [480, 270]);
        assert_eq!(Film::new(1919, 1081, 3).size, [640, 361]);
        assert_eq!(Film::new(5, 5, 0).size, [5, 5]);
        assert!(Film::new(0, 10, 4).is_empty());
    }

    #[test]
    fn upscale_mapping_is_exact() {
        for (w, h, s) in [(640, 360, 4), (643, 357, 4), (101, 77, 3), (9, 9, 8)] {
            let film = Film::new(w, h, s);
            let covered = film.covered();
            assert!(covered[0] >= w && covered[0] < w + s);
            assert!(covered[1] >= h && covered[1] < h + s);
            // Each film pixel shows on exactly s×s view pixels of the covered
            // area, all in one block.
            let mut count = vec![0u32; (film.size[0] * film.size[1]) as usize];
            for y in 0..covered[1] {
                for x in 0..covered[0] {
                    let [i, j] = film.film_pixel(x, y);
                    assert!(i < film.size[0] && j < film.size[1]);
                    assert_eq!((x / s, y / s), (i, j));
                    count[(j * film.size[0] + i) as usize] += 1;
                }
            }
            assert!(count.iter().all(|&c| c == s * s));
        }
    }

    #[test]
    fn grid_is_anchored_to_the_view() {
        // Growing the view keeps every existing view pixel on the same film
        // pixel: the grid starts at the top-left corner, not the centre.
        let a = Film::new(100, 60, 4);
        let b = Film::new(103, 61, 4);
        for y in 0..60 {
            for x in 0..100 {
                assert_eq!(a.film_pixel(x, y), b.film_pixel(x, y));
            }
        }
    }
}
