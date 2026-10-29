//! The camera as the film sees it: its world frame and the off-centre
//! frustum (or box) that puts film pixel edges on view pixel multiples of
//! `pixel_size`.
//!
//! Positions on the film are in "film space": for a perspective camera the
//! tangent of a ray's angle (a camera-space ray `(tx, ty, -1)`), for an
//! orthographic one a length in scene units across the camera's plane (a
//! ray from `(tx, ty, 0)` down -Z). The view's own window (the camera's
//! horizontal aperture across the view's width) is kept, and the film's is
//! that window extended right and down to the film's covered area, so a
//! point seen at view pixel `x` lands at film coordinate `x / pixel_size`
//! exactly, whichever the projection.
//!
//! Depth is reversed (1 at the near plane, 0 at the far) for the
//! G-buffer's float depth test: a perspective far plane may be infinite;
//! an orthographic one is drawn in to the scene's bounds, so its linear
//! depth keeps its precision.

use crate::math::{Mat4, Vec3, Vec4};

use crate::film::Film;
use crate::scene::{Camera, Projection};

/// Far distances past this are infinite.
const INFINITE: f64 = 1e30;

/// The camera's world frame and film window, ready for the GPU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraFrame {
    /// The projection.
    pub projection: Projection,
    /// The eye (perspective) or the centre of the camera's plane
    /// (orthographic), in world space.
    pub position: Vec3,
    /// Camera +X, +Y and +Z (the camera looks down -Z), unit length.
    pub axes: [Vec3; 3],
    /// Film-space coordinates of the film's top-left corner.
    pub origin: [f64; 2],
    /// Film-space extent of one film pixel (square).
    pub step: f64,
    /// The film's size, in film pixels.
    pub size: [u32; 2],
    /// Near and far clip distances along the view direction (far may be
    /// infinite for a perspective camera).
    pub clip: [f64; 2],
}

impl CameraFrame {
    /// The frame of `camera` for `film`. `bounds` (the scene's, world
    /// space) draws an orthographic camera's far plane in to the scene.
    pub fn new(camera: &Camera, film: &Film, bounds: Option<([f32; 3], [f32; 3])>) -> CameraFrame {
        let t = &camera.transform;
        let col = |i: usize| Vec3::new(t[i][0] as f64, t[i][1] as f64, t[i][2] as f64);
        // Orthonormalize: the camera's scale is not a lens.
        let back = col(2).normalize_or(Vec3::Z);
        let right = col(1).cross(back).normalize_or(Vec3::X);
        let up = back.cross(right);
        let position = col(3);
        let view_w = film.view[0].max(1) as f64;
        let view_h = film.view[1].max(1) as f64;
        // Film space per millimetre of the film back: a tangent over the
        // focal length, or a length over the scene's scale.
        let per_mm = match camera.projection {
            Projection::Perspective => 1.0 / (camera.focal_length as f64).max(1e-6),
            Projection::Orthographic => 1.0 / (camera.mm_per_unit as f64).max(1e-9),
        };
        // The aperture fits the view horizontally.
        let per_view_pixel = camera.horizontal_aperture as f64 * per_mm / view_w;
        let shift = [
            camera.horizontal_aperture_offset as f64 * per_mm,
            camera.vertical_aperture_offset as f64 * per_mm,
        ];
        let near = (camera.clipping_range[0] as f64).max(1e-6);
        let mut far = (camera.clipping_range[1] as f64).max(near * 2.0);
        if camera.projection == Projection::Orthographic {
            // An orthographic depth is linear: the far plane at the scene.
            let deepest = bounds.map_or(near * 2.0, |(lo, hi)| {
                (0..8)
                    .map(|k| {
                        let pick = |a: usize| if k >> a & 1 == 0 { lo[a] } else { hi[a] };
                        let p = Vec3::new(pick(0) as f64, pick(1) as f64, pick(2) as f64);
                        -(p - position).dot(back)
                    })
                    .fold(f64::MIN, f64::max)
            });
            far = far.min(deepest * 1.01 + 1e-3).max(near * 2.0);
        }
        CameraFrame {
            projection: camera.projection,
            position,
            axes: [right, up, back],
            origin: [
                -0.5 * view_w * per_view_pixel + shift[0],
                0.5 * view_h * per_view_pixel + shift[1],
            ],
            step: per_view_pixel * film.pixel_size as f64,
            size: film.size,
            clip: [near, far],
        }
    }

    /// Whether the camera is orthographic.
    pub fn orthographic(&self) -> bool {
        self.projection == Projection::Orthographic
    }

    /// Film coordinate (`fx`, `fy`), in film pixels from the top-left
    /// corner (pixel centres at +½), in film space.
    fn film_space(&self, fx: f64, fy: f64) -> [f64; 2] {
        [
            self.origin[0] + fx * self.step,
            self.origin[1] - fy * self.step,
        ]
    }

    /// The world ray through film coordinate (`fx`, `fy`): its origin and
    /// its direction (not unit; one unit of depth a unit of it).
    pub fn ray(&self, fx: f64, fy: f64) -> (Vec3, Vec3) {
        let [tx, ty] = self.film_space(fx, fy);
        let [right, up, back] = self.axes;
        if self.orthographic() {
            (self.position + right * tx + up * ty, -back)
        } else {
            (self.position, right * tx + up * ty - back)
        }
    }

    /// Film coordinates of a world point, and its depth along the view
    /// direction; none behind the eye (or the orthographic camera's plane).
    pub fn project(&self, p: Vec3) -> Option<([f64; 2], f64)> {
        let d = p - self.position;
        let depth = -d.dot(self.axes[2]);
        if depth <= 0.0 {
            return None;
        }
        let scale = if self.orthographic() { 1.0 } else { depth };
        let tx = d.dot(self.axes[0]) / scale;
        let ty = d.dot(self.axes[1]) / scale;
        Some((
            [
                (tx - self.origin[0]) / self.step,
                (self.origin[1] - ty) / self.step,
            ],
            depth,
        ))
    }

    /// World to clip space over the film window, depth reversed (wgpu's
    /// range: 1 at the near plane, 0 at the far).
    pub fn view_projection(&self) -> Mat4 {
        let [r, u, b] = self.axes;
        let e = self.position;
        let view = Mat4::from_cols(
            Vec4::new(r.x, u.x, b.x, 0.0),
            Vec4::new(r.y, u.y, b.y, 0.0),
            Vec4::new(r.z, u.z, b.z, 0.0),
            Vec4::new(-r.dot(e), -u.dot(e), -b.dot(e), 1.0),
        );
        let left = self.origin[0];
        let right = left + self.size[0].max(1) as f64 * self.step;
        let top = self.origin[1];
        let bottom = top - self.size[1].max(1) as f64 * self.step;
        let (sx, sy) = (2.0 / (right - left), 2.0 / (top - bottom));
        let (cx, cy) = (
            (right + left) / (right - left),
            (top + bottom) / (top - bottom),
        );
        let [n, f] = self.clip;
        let proj = if self.orthographic() {
            // z = (f - depth) / (f - n), w = 1 (camera space z is -depth).
            Mat4::from_cols(
                Vec4::new(sx, 0.0, 0.0, 0.0),
                Vec4::new(0.0, sy, 0.0, 0.0),
                Vec4::new(0.0, 0.0, 1.0 / (f - n), 0.0),
                Vec4::new(-cx, -cy, f / (f - n), 1.0),
            )
        } else {
            // z = n (f - depth) / (depth (f - n)), w = depth; n / depth
            // when the far plane is infinite.
            let (a, c) = if f >= INFINITE {
                (0.0, n)
            } else {
                (n / (f - n), n * f / (f - n))
            };
            Mat4::from_cols(
                Vec4::new(sx, 0.0, 0.0, 0.0),
                Vec4::new(0.0, sy, 0.0, 0.0),
                Vec4::new(cx, cy, a, -1.0),
                Vec4::new(0.0, 0.0, c, 0.0),
            )
        };
        proj * view
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Matrix;

    /// A camera at (1, 2, 6), turned a little, looking roughly at the origin.
    fn camera(projection: Projection) -> Camera {
        let m = Mat4::from_rotation_y(0.3) * Mat4::from_rotation_x(-0.2);
        let mut t: Matrix = [[0.0; 4]; 4];
        for (i, c) in [m.col(0), m.col(1), m.col(2)].iter().enumerate() {
            t[i] = [c.x as f32, c.y as f32, c.z as f32, 0.0];
        }
        t[3] = [1.0, 2.0, 6.0, 1.0];
        Camera {
            projection,
            // Orthographic: 4 units across at 1000 mm a unit.
            horizontal_aperture: if projection == Projection::Orthographic {
                4000.0
            } else {
                36.0
            },
            horizontal_aperture_offset: 1.5,
            vertical_aperture_offset: -0.7,
            transform: t,
            ..Camera::default()
        }
    }

    const BOTH: [Projection; 2] = [Projection::Perspective, Projection::Orthographic];
    type Bounds = Option<([f32; 3], [f32; 3])>;
    const BOUNDS: Bounds = Some(([-2.0, -1.0, -1.5], [2.0, 1.5, 1.5]));

    /// Points spread through the frustum.
    fn points() -> Vec<Vec3> {
        let mut out = Vec::new();
        for i in 0..7 {
            for j in 0..5 {
                for k in 0..3 {
                    out.push(Vec3::new(
                        i as f64 * 0.5 - 1.5,
                        j as f64 * 0.4 - 0.8,
                        k as f64 * 1.0 - 1.0,
                    ));
                }
            }
        }
        out
    }

    #[test]
    fn film_coordinates_are_view_coordinates_over_pixel_size() {
        // The grid lock: what the full-resolution view shows at view pixel x
        // shows at film coordinate x / s, for any view size (the extra
        // ceil'd column extends the window and never rescales it).
        for projection in BOTH {
            let cam = camera(projection);
            for (w, h, s) in [(640, 360, 4), (643, 357, 4), (101, 77, 3)] {
                let full = CameraFrame::new(&cam, &Film::new(w, h, 1), BOUNDS);
                let film = CameraFrame::new(&cam, &Film::new(w, h, s), BOUNDS);
                for p in points() {
                    let (a, da) = full.project(p).expect("in front");
                    let (b, db) = film.project(p).expect("in front");
                    assert!((a[0] / s as f64 - b[0]).abs() < 1e-9, "{a:?} {b:?}");
                    assert!((a[1] / s as f64 - b[1]).abs() < 1e-9, "{a:?} {b:?}");
                    assert!((da - db).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn rays_and_projection_agree() {
        for projection in BOTH {
            let frame = CameraFrame::new(&camera(projection), &Film::new(640, 360, 4), BOUNDS);
            for (fx, fy) in [(0.5, 0.5), (80.0, 45.0), (159.5, 89.5), (13.25, 70.75)] {
                let (o, d) = frame.ray(fx, fy);
                let (f, depth) = frame.project(o + d * 3.0).expect("in front");
                assert!((f[0] - fx).abs() < 1e-9 && (f[1] - fy).abs() < 1e-9);
                assert!((depth - 3.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn orthographic_rays_are_parallel_and_the_aperture_is_the_width() {
        let film = Film::new(640, 360, 4);
        let frame = CameraFrame::new(&camera(Projection::Orthographic), &film, BOUNDS);
        let (o0, d0) = frame.ray(0.0, 0.0);
        let (o1, d1) = frame.ray(160.0, 0.0);
        assert_eq!(d0, d1);
        // 4000 mm at 1000 mm a unit: 4 units across the view.
        assert!(((o1 - o0).length() - 4.0).abs() < 1e-9);
        // A point moved along the view direction keeps its film position.
        let p = Vec3::new(0.3, 0.2, 0.1);
        let (a, _) = frame.project(p).unwrap();
        let (b, _) = frame.project(p + d0 * 2.0).unwrap();
        assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
        // The far plane is drawn in to the bounds.
        assert!(frame.clip[1] < 20.0, "far {}", frame.clip[1]);
    }

    #[test]
    fn clip_space_matches_film_coordinates_with_depth_reversed() {
        let film = Film::new(643, 357, 4);
        for projection in BOTH {
            let frame = CameraFrame::new(&camera(projection), &film, BOUNDS);
            let vp = frame.view_projection();
            let mut by_depth = Vec::new();
            for p in points() {
                let (f, depth) = frame.project(p).expect("in front");
                let c = vp.mul_vec4(Vec4::new(p.x, p.y, p.z, 1.0));
                let ndc = [c.x / c.w, c.y / c.w, c.z / c.w];
                let fx = (ndc[0] + 1.0) * 0.5 * film.size[0] as f64;
                let fy = (1.0 - ndc[1]) * 0.5 * film.size[1] as f64;
                assert!((fx - f[0]).abs() < 1e-6 && (fy - f[1]).abs() < 1e-6);
                assert!(ndc[2] > 0.0 && ndc[2] < 1.0, "{projection:?} z {}", ndc[2]);
                by_depth.push((depth, ndc[2]));
            }
            // Nearer is larger.
            by_depth.sort_by(|a, b| a.0.total_cmp(&b.0));
            assert!(by_depth.windows(2).all(|w| w[0].1 >= w[1].1));
        }
    }

    #[test]
    fn the_view_centre_is_the_optical_axis() {
        // Without aperture shifts the view's centre (not the film's) looks
        // straight down -Z (orthographic: from the camera's position).
        for projection in BOTH {
            let cam = Camera {
                projection,
                ..Camera::default()
            };
            let film = Film::new(643, 357, 4);
            let frame = CameraFrame::new(&cam, &film, None);
            let centre = [643.0 / 2.0 / 4.0, 357.0 / 2.0 / 4.0];
            let (o, d) = frame.ray(centre[0], centre[1]);
            assert!(d.x.abs() < 1e-12 && d.y.abs() < 1e-12 && d.z == -1.0);
            assert!(o.length() < 1e-12);
        }
    }
}
