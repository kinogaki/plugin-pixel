//! The camera as the film sees it: its world frame and the off-centre
//! frustum (or box) that puts film pixel edges on view pixel multiples of
//! the pixel size.
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
//! G-buffer's float depth test: a perspective far plane may be infinite; an
//! orthographic one is drawn in to the scene's bounds, so its linear depth
//! keeps its precision.
use super::film::Film;

/// Far distances past this are infinite.
const INFINITE: f64 = 1e30;

/// A camera after UsdGeomCamera, looking down its local -Z with +Y up, its
/// film back fitting the view horizontally.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// Parallel rays rather than rays from the eye.
    pub orthographic: bool,
    /// The lens's focal length, mm.
    pub focal_length: f32,
    /// The film back's width, mm (orthographic: the view's width in mm on
    /// the scene's scale).
    pub aperture: f32,
    /// The film back's shift, mm.
    pub offset: [f32; 2],
    /// Near and far clipping distances, scene units.
    pub clip: [f32; 2],
    /// Millimetres in a scene unit: what turns an orthographic aperture into
    /// scene units.
    pub mm_per_unit: f32,
    /// Camera to world: three rows of a 3 × 4 affine matrix.
    pub rows: [[f64; 4]; 3],
}

impl Default for Camera {
    /// A 50 mm lens on a 36 mm back at the origin, metres.
    fn default() -> Camera {
        Camera {
            orthographic: false,
            focal_length: 50.0,
            aperture: 36.0,
            offset: [0.0; 2],
            clip: [0.1, 10_000.0],
            mm_per_unit: 1000.0,
            rows: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        }
    }
}

type D3 = [f64; 3];

fn dot(a: D3, b: D3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: D3, b: D3) -> D3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit_or(a: D3, fallback: D3) -> D3 {
    let l = dot(a, a).sqrt();
    if l > 1e-12 && l.is_finite() {
        a.map(|v| v / l)
    } else {
        fallback
    }
}

fn axpy(a: D3, s: f64, b: D3) -> D3 {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]
}

/// The camera's world frame and film window, ready for the GPU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraFrame {
    /// Parallel rays.
    pub orthographic: bool,
    /// The eye (perspective) or the centre of the camera's plane
    /// (orthographic), world space.
    pub position: D3,
    /// Camera +X, +Y and +Z (it looks down -Z), unit length.
    pub axes: [D3; 3],
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
        let r = &camera.rows;
        let col = |i: usize| [r[0][i], r[1][i], r[2][i]];
        // Orthonormalize: the camera's scale is not a lens.
        let back = unit_or(col(2), [0.0, 0.0, 1.0]);
        let right = unit_or(cross(col(1), back), [1.0, 0.0, 0.0]);
        let up = cross(back, right);
        let position = col(3);
        let [view_w, view_h] = film.view.map(|n| f64::from(n.max(1)));
        // Film space a millimetre of the film back: a tangent over the focal
        // length, or a length over the scene's scale.
        let per_mm = match camera.orthographic {
            false => 1.0 / f64::from(camera.focal_length).max(1e-6),
            true => 1.0 / f64::from(camera.mm_per_unit).max(1e-9),
        };
        let per_view_pixel = f64::from(camera.aperture) * per_mm / view_w;
        let shift = camera.offset.map(|o| f64::from(o) * per_mm);
        let near = f64::from(camera.clip[0]).max(1e-6);
        let mut far = f64::from(camera.clip[1]).max(near * 2.0);
        if camera.orthographic {
            // An orthographic depth is linear: the far plane at the scene.
            let deepest = bounds.map_or(near * 2.0, |(lo, hi)| {
                (0..8)
                    .map(|k| {
                        let pick =
                            |a: usize| f64::from(if k >> a & 1 == 0 { lo[a] } else { hi[a] });
                        let p = [pick(0), pick(1), pick(2)];
                        -dot(axpy(p, -1.0, position), back)
                    })
                    .fold(f64::MIN, f64::max)
            });
            far = far.min(deepest * 1.01 + 1e-3).max(near * 2.0);
        }
        CameraFrame {
            orthographic: camera.orthographic,
            position,
            axes: [right, up, back],
            origin: [
                -0.5 * view_w * per_view_pixel + shift[0],
                0.5 * view_h * per_view_pixel + shift[1],
            ],
            step: per_view_pixel * f64::from(film.pixel_size),
            size: film.size,
            clip: [near, far],
        }
    }

    /// The world ray through film coordinate (`fx`, `fy`), in film pixels
    /// from the top-left corner (pixel centres at +½): its origin and its
    /// direction (not unit; one unit of depth a unit of it).
    pub fn ray(&self, fx: f64, fy: f64) -> (D3, D3) {
        let tx = self.origin[0] + fx * self.step;
        let ty = self.origin[1] - fy * self.step;
        let [right, up, back] = self.axes;
        let across = axpy(right.map(|v| v * tx), ty, up);
        match self.orthographic {
            true => (axpy(self.position, 1.0, across), back.map(|v| -v)),
            false => (self.position, axpy(across, -1.0, back)),
        }
    }

    /// Film coordinates of a world point, and its depth along the view
    /// direction; none behind the eye (or the orthographic camera's plane).
    pub fn project(&self, p: D3) -> Option<([f64; 2], f64)> {
        let d = axpy(p, -1.0, self.position);
        let depth = -dot(d, self.axes[2]);
        if depth <= 0.0 {
            return None;
        }
        let scale = if self.orthographic { 1.0 } else { depth };
        let tx = dot(d, self.axes[0]) / scale;
        let ty = dot(d, self.axes[1]) / scale;
        let film = [
            (tx - self.origin[0]) / self.step,
            (self.origin[1] - ty) / self.step,
        ];
        Some((film, depth))
    }

    /// World to clip space over the film window, depth reversed (1 at the
    /// near plane, 0 at the far): four columns, as WGSL's `mat4x4` reads
    /// them.
    pub fn view_projection(&self) -> [[f32; 4]; 4] {
        let [r, u, b] = self.axes;
        let e = self.position;
        // Rows of the view matrix: world to camera space.
        let view = [
            [r[0], r[1], r[2], -dot(r, e)],
            [u[0], u[1], u[2], -dot(u, e)],
            [b[0], b[1], b[2], -dot(b, e)],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let left = self.origin[0];
        let right = left + f64::from(self.size[0].max(1)) * self.step;
        let top = self.origin[1];
        let bottom = top - f64::from(self.size[1].max(1)) * self.step;
        let (sx, sy) = (2.0 / (right - left), 2.0 / (top - bottom));
        let (cx, cy) = (
            (right + left) / (right - left),
            (top + bottom) / (top - bottom),
        );
        let [n, f] = self.clip;
        // Rows of the projection (camera space z is -depth).
        let projection = if self.orthographic {
            // z = (f - depth) / (f - n), w = 1.
            [
                [sx, 0.0, 0.0, -cx],
                [0.0, sy, 0.0, -cy],
                [0.0, 0.0, 1.0 / (f - n), f / (f - n)],
                [0.0, 0.0, 0.0, 1.0],
            ]
        } else {
            // z = n (f - depth) / (depth (f - n)), w = depth; n / depth when
            // the far plane is infinite.
            let (a, c) = match f >= INFINITE {
                true => (0.0, n),
                false => (n / (f - n), n * f / (f - n)),
            };
            [
                [sx, 0.0, cx, 0.0],
                [0.0, sy, cy, 0.0],
                [0.0, 0.0, a, c],
                [0.0, 0.0, -1.0, 0.0],
            ]
        };
        // Column j of projection × view.
        std::array::from_fn(|j| {
            std::array::from_fn(|i| {
                (0..4).map(|k| projection[i][k] * view[k][j]).sum::<f64>() as f32
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera at (1, 2, 6), turned a little, looking roughly at the origin.
    fn camera(orthographic: bool) -> Camera {
        let (sy, cy) = 0.3f64.sin_cos();
        let (sx, cx) = (-0.2f64).sin_cos();
        // Ry × Rx, as rows.
        let rows = [
            [cy, sy * sx, sy * cx, 1.0],
            [0.0, cx, -sx, 2.0],
            [-sy, cy * sx, cy * cx, 6.0],
        ];
        Camera {
            orthographic,
            // Orthographic: 4 units across at 1000 mm a unit.
            aperture: if orthographic { 4000.0 } else { 36.0 },
            offset: [1.5, -0.7],
            rows,
            ..Camera::default()
        }
    }

    const BOTH: [bool; 2] = [false, true];
    const BOUNDS: Option<([f32; 3], [f32; 3])> = Some(([-2.0, -1.0, -1.5], [2.0, 1.5, 1.5]));

    /// Points spread through the frustum.
    fn points() -> Vec<D3> {
        let mut out = Vec::new();
        for i in 0..7 {
            for j in 0..5 {
                for k in 0..3 {
                    out.push([
                        f64::from(i) * 0.5 - 1.5,
                        f64::from(j) * 0.4 - 0.8,
                        f64::from(k) - 1.0,
                    ]);
                }
            }
        }
        out
    }

    #[test]
    fn film_coordinates_are_view_coordinates_over_pixel_size() {
        // The grid lock: what the full-resolution view shows at view pixel x
        // shows at film coordinate x / s, for any view size.
        for orthographic in BOTH {
            let cam = camera(orthographic);
            for (w, h, s) in [(640, 360, 4), (643, 357, 4), (101, 77, 3)] {
                let full = CameraFrame::new(&cam, &Film::new(w, h, 1), BOUNDS);
                let film = CameraFrame::new(&cam, &Film::new(w, h, s), BOUNDS);
                for p in points() {
                    let (a, da) = full.project(p).expect("in front");
                    let (b, db) = film.project(p).expect("in front");
                    let s = f64::from(s);
                    assert!((a[0] / s - b[0]).abs() < 1e-9 && (a[1] / s - b[1]).abs() < 1e-9);
                    assert!((da - db).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn rays_and_projection_agree() {
        for orthographic in BOTH {
            let frame = CameraFrame::new(&camera(orthographic), &Film::new(640, 360, 4), BOUNDS);
            for (fx, fy) in [(0.5, 0.5), (80.0, 45.0), (159.5, 89.5), (13.25, 70.75)] {
                let (o, d) = frame.ray(fx, fy);
                let (f, depth) = frame.project(axpy(o, 3.0, d)).expect("in front");
                assert!((f[0] - fx).abs() < 1e-9 && (f[1] - fy).abs() < 1e-9);
                assert!((depth - 3.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn orthographic_rays_are_parallel_and_the_aperture_is_the_width() {
        let frame = CameraFrame::new(&camera(true), &Film::new(640, 360, 4), BOUNDS);
        let (o0, d0) = frame.ray(0.0, 0.0);
        let (o1, d1) = frame.ray(160.0, 0.0);
        assert_eq!(d0, d1);
        // 4000 mm at 1000 mm a unit: 4 units across the view.
        let across = axpy(o1, -1.0, o0);
        assert!((dot(across, across).sqrt() - 4.0).abs() < 1e-9);
        // The far plane is drawn in to the bounds.
        assert!(frame.clip[1] < 20.0, "far {}", frame.clip[1]);
    }

    #[test]
    fn clip_space_matches_film_coordinates_with_depth_reversed() {
        let film = Film::new(643, 357, 4);
        for orthographic in BOTH {
            let frame = CameraFrame::new(&camera(orthographic), &film, BOUNDS);
            let m = frame.view_projection();
            let mut by_depth = Vec::new();
            for p in points() {
                let (f, depth) = frame.project(p).expect("in front");
                let c: [f64; 4] = std::array::from_fn(|i| {
                    let column = |j: usize| f64::from(m[j][i]);
                    column(0) * p[0] + column(1) * p[1] + column(2) * p[2] + column(3)
                });
                let ndc = [c[0] / c[3], c[1] / c[3], c[2] / c[3]];
                let fx = (ndc[0] + 1.0) * 0.5 * f64::from(film.size[0]);
                let fy = (1.0 - ndc[1]) * 0.5 * f64::from(film.size[1]);
                assert!((fx - f[0]).abs() < 1e-3 && (fy - f[1]).abs() < 1e-3);
                assert!(ndc[2] > 0.0 && ndc[2] < 1.0, "z {}", ndc[2]);
                by_depth.push((depth, ndc[2]));
            }
            // Nearer is larger.
            by_depth.sort_by(|a, b| a.0.total_cmp(&b.0));
            assert!(by_depth.windows(2).all(|w| w[0].1 >= w[1].1));
        }
    }

    #[test]
    fn the_view_centre_is_the_optical_axis() {
        for orthographic in BOTH {
            let cam = Camera {
                orthographic,
                ..Camera::default()
            };
            let frame = CameraFrame::new(&cam, &Film::new(643, 357, 4), None);
            let (o, d) = frame.ray(643.0 / 8.0, 357.0 / 8.0);
            assert!(d[0].abs() < 1e-12 && d[1].abs() < 1e-12 && d[2] == -1.0);
            assert!(dot(o, o) < 1e-24);
        }
    }
}
