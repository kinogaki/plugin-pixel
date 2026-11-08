//! Lights as Pixel renders them: each a point at its centre (or parallel
//! light) of the power UsdLux gives its shape, shaped by ShapingAPI (a
//! cone, focus, an IES profile read around its axis), shadowed as ShadowAPI
//! says; and the shapes the camera sees ([`Glow`]), drawn unlit and traced
//! by no ray.
use super::mesh::{Corners, Mesh};
use std::f32::consts::PI;
use std::sync::{Arc, LazyLock};

/// A light with a place or a direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    /// Its shape.
    pub shape: LightShape,
    /// Its radiance: the emission's color × intensity × 2^exposure × its
    /// Diffuse (Pixel shades diffuse light alone) × a color map's mean.
    pub radiance: [f32; 3],
    /// Its power does not change with its size.
    pub normalize: bool,
    /// Light to world: three rows (it shines down its -Z).
    pub rows: [[f32; 4]; 3],
    /// Its cookie, on distant and sphere lights.
    pub gobo: Option<Gobo>,
    /// ShapingAPI's shaping.
    pub shaping: Option<Shaping>,
    /// ShadowAPI's shadows.
    pub shadow: Shadow,
}

impl Light {
    /// The area it shines from in world space (its rows' scale in it; a
    /// point's and a line's their shape's), which a normalized light's
    /// power spreads over: none for a distant light.
    pub fn area(&self) -> Option<f32> {
        let axis = |k: usize| self.rows.iter().map(|r| r[k] * r[k]).sum::<f32>().sqrt();
        let [x, y, z] = [0, 1, 2].map(axis);
        match self.shape {
            LightShape::Area { area } => Some(area * x * y),
            LightShape::Distant { .. } => None,
            LightShape::Sphere { radius, .. } => {
                let r = radius * x.max(y).max(z);
                Some(4.0 * PI * r * r)
            }
            LightShape::Cylinder { length, radius, .. } => {
                Some(2.0 * PI * radius * y.max(z) * length * x)
            }
        }
    }
}

/// A light's shape, as UsdLux shapes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightShape {
    /// A rectangle or disk of `area` in its XY plane, shining down -Z.
    Area {
        /// Its area in its own units.
        area: f32,
    },
    /// Parallel light `angle` degrees across.
    Distant {
        /// The sun's size across.
        angle: f32,
    },
    /// A sphere of `radius`, or a point of its power.
    Sphere {
        /// Its radius.
        radius: f32,
        /// Treated as a point.
        point: bool,
    },
    /// A tube `length` along its X of `radius`, or a line of its power.
    Cylinder {
        /// Along X.
        length: f32,
        /// Its radius.
        radius: f32,
        /// Treated as a line.
        line: bool,
    },
}

/// Vertical angles an IES profile is read at, from the light's axis.
pub const PROFILE: usize = 16;

/// ShapingAPI: what a light's light is multiplied by toward a direction,
/// read against its -Z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shaping {
    /// The cone's angle, radians (π: none).
    pub cone: f32,
    /// Where its softness starts, radians.
    pub start: f32,
    /// `|cos|^focus` toward the axis (0: none).
    pub focus: f32,
    /// What off-axis light fades to as it focuses.
    pub tint: [f32; 3],
    /// The IES profile's candela at [`PROFILE`] vertical angles ((k + ½) /
    /// PROFILE × 180°), the mean around its axis (over its power when it
    /// normalizes); none: no profile.
    pub profile: Option<[f32; PROFILE]>,
    /// The profile's angle scale.
    pub angle_scale: f32,
}

/// ShadowAPI: what a light's shadow rays take as shadow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shadow {
    /// What shadowed light is multiplied by.
    pub color: [f32; 3],
    /// How far an occluder may be (−1: any; 0: nothing shadows).
    pub reach: f32,
    /// How far before that the shadow fades out (−1: none).
    pub falloff: f32,
    /// The fade's gamma.
    pub gamma: f32,
}

impl Default for Shadow {
    /// Black shadows at any distance.
    fn default() -> Shadow {
        Shadow {
            color: [0.0; 3],
            reach: -1.0,
            falloff: -1.0,
            gamma: 1.0,
        }
    }
}

/// A mask of dappled light on a distant light or sphere, as if through
/// leaves out of frame: value-noise fBm on the light's plane, `scale` scene
/// units a feature, its edges `contrast` hard.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gobo {
    /// The pattern's features' size, scene units.
    pub scale: f32,
    /// 0 a soft mottle, 1 hard light and shade.
    pub contrast: f32,
    /// Picks the pattern.
    pub seed: u32,
    /// Slides the pattern across the plane, scene units.
    pub offset: [f32; 2],
}

/// A light's shape the camera sees (its Visible to Camera on): its mesh
/// placed and sized, shining its radiance (over its area when it
/// normalizes), drawn unlit and traced by no ray. A rect's and a disk's show
/// from both sides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glow {
    /// Which of [`glow_meshes`] it is.
    pub mesh: GlowMesh,
    /// The mesh to world: three rows, its size in them.
    pub to_world: [[f32; 4]; 3],
    /// What it shines.
    pub radiance: [f32; 3],
}

/// The unit shapes glows are made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlowMesh {
    /// A 1 × 1 square in XY facing -Z.
    Rect = 0,
    /// A disk of radius ½ in XY facing -Z.
    Disk = 1,
    /// A sphere of radius ½.
    Sphere = 2,
    /// A tube 1 along X of radius ½, open at its ends.
    Tube = 3,
}

/// Segments around a glow's circle.
const AROUND: u32 = 32;

/// The unit glow meshes, in [`GlowMesh`]'s order, made once.
pub fn glow_meshes() -> &'static [Arc<Mesh>; 4] {
    static MESHES: LazyLock<[Arc<Mesh>; 4]> = LazyLock::new(|| {
        [rect(), disk(), sphere(), tube()].map(|(points, triangles, normals)| {
            let corners = Corners {
                normals: Some(normals),
                ..Corners::default()
            };
            Arc::new(Mesh::new(&points, &triangles, corners))
        })
    });
    &MESHES
}

/// A mesh's points, triangles and a normal each corner.
type Made = (Vec<[f32; 3]>, Vec<[u32; 3]>, Vec<[f32; 3]>);

/// Each corner's normal, `normal` of its point.
fn normals(
    points: &[[f32; 3]],
    triangles: &[[u32; 3]],
    normal: impl Fn([f32; 3]) -> [f32; 3],
) -> Vec<[f32; 3]> {
    (triangles.iter().flatten())
        .map(|&i| normal(points[i as usize]))
        .collect()
}

fn rect() -> Made {
    let points = vec![
        [-0.5, -0.5, 0.0],
        [-0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
    ];
    // Wound to face -Z.
    let triangles = vec![[0, 1, 2], [0, 2, 3]];
    let n = normals(&points, &triangles, |_| [0.0, 0.0, -1.0]);
    (points, triangles, n)
}

fn disk() -> Made {
    let mut points = vec![[0.0, 0.0, 0.0]];
    points.extend((0..AROUND).map(|k| {
        let a = k as f32 / AROUND as f32 * 2.0 * PI;
        [0.5 * a.cos(), 0.5 * a.sin(), 0.0]
    }));
    let triangles: Vec<[u32; 3]> = (0..AROUND)
        .map(|k| [0, 1 + (k + 1) % AROUND, 1 + k])
        .collect();
    let n = normals(&points, &triangles, |_| [0.0, 0.0, -1.0]);
    (points, triangles, n)
}

fn sphere() -> Made {
    let rings = AROUND / 2;
    let mut points = Vec::new();
    for ring in 0..=rings {
        let theta = ring as f32 / rings as f32 * PI;
        for k in 0..AROUND {
            let phi = k as f32 / AROUND as f32 * 2.0 * PI;
            let s = theta.sin();
            points.push([0.5 * s * phi.cos(), 0.5 * theta.cos(), 0.5 * s * phi.sin()]);
        }
    }
    let at = |ring: u32, k: u32| ring * AROUND + k % AROUND;
    let mut triangles = Vec::new();
    for ring in 0..rings {
        for k in 0..AROUND {
            let (a, b, c, d) = (
                at(ring, k),
                at(ring, k + 1),
                at(ring + 1, k),
                at(ring + 1, k + 1),
            );
            triangles.extend([[a, b, d], [a, d, c]]);
        }
    }
    let n = normals(&points, &triangles, |p| p.map(|v| 2.0 * v));
    (points, triangles, n)
}

fn tube() -> Made {
    let mut points = Vec::new();
    for x in [-0.5, 0.5] {
        points.extend((0..AROUND).map(|k| {
            let a = k as f32 / AROUND as f32 * 2.0 * PI;
            [x, 0.5 * a.cos(), 0.5 * a.sin()]
        }));
    }
    let mut triangles = Vec::new();
    for k in 0..AROUND {
        let (a, b) = (k, (k + 1) % AROUND);
        triangles.extend([[a, b, AROUND + b], [a, AROUND + b, AROUND + k]]);
    }
    let n = normals(&points, &triangles, |p| [0.0, 2.0 * p[1], 2.0 * p[2]]);
    (points, triangles, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glow_meshes_are_unit_shapes_their_normals_outward() {
        let [rect, disk, sphere, tube] = glow_meshes().each_ref();
        assert_eq!(
            (rect.triangles.len(), disk.triangles.len()),
            (2, AROUND as usize)
        );
        // The square faces -Z: its winding's normal points down Z.
        let [a, b, c] = rect.triangles[0].map(|i| rect.positions[i as usize]);
        let (u, v) = (
            [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
            [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
        );
        assert!(u[0] * v[1] - u[1] * v[0] < 0.0, "wound to face -Z");
        let far = |m: &Mesh| {
            (m.positions.iter())
                .map(|p| p.iter().map(|v| v.abs()).fold(0.0, f32::max))
                .fold(0.0, f32::max)
        };
        assert!((far(sphere) - 0.5).abs() < 1e-6 && (far(tube) - 0.5).abs() < 1e-6);
    }
}
