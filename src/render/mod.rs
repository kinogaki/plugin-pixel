//! Pixel's renderer, knowing no SDK: the scene it renders ([`Scene`]: meshes
//! in their own space placed by instances, the rest in world space), the
//! film and camera ([`film`], [`camera`]), the two levels of BVH its rays
//! trace ([`bvh`]) and the dither ([`dither`]). The passes that render it on a
//! wgpu device are the crate's `gpu` module; `sync` makes a [`Scene`] of
//! the host's prims.
pub mod bvh;
pub mod camera;
pub mod dither;
pub mod film;
mod light;
mod mesh;
pub mod settings;
pub mod vector;

pub use light::{Glow, GlowMesh, Gobo, Light, LightShape, PROFILE, Shadow, Shaping, glow_meshes};
pub use mesh::{Corners, Mesh};

use std::sync::Arc;

/// Everything a frame renders, in linear ACEScg.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    /// The surfaces' geometry, each in its own space, shared by the
    /// instances placing it.
    pub meshes: Vec<Arc<Mesh>>,
    /// The meshes placed: what is drawn and traced.
    pub instances: Vec<Instance>,
    /// The materials instances shade with, by [`Instance::surface`].
    pub surfaces: Vec<Surface>,
    /// The pictures surfaces sample, by [`Texture::picture`].
    pub pictures: Vec<Arc<Picture>>,
    /// The closed meshes filled with a medium (never drawn).
    pub volumes: Vec<Volume>,
    /// The lights with a place or a direction.
    pub lights: Vec<Light>,
    /// The lights' shapes the camera sees.
    pub glows: Vec<Glow>,
    /// The domes' light.
    pub ambient: Ambient,
    /// The portals domes light through.
    pub portals: Vec<Portal>,
    /// The rim lights' light.
    pub rims: Rims,
    /// The height fog, when there is one.
    pub fog: Option<Fog>,
}

/// A picture a surface samples: linear ACEScg texels, straight (not
/// premultiplied), rows from the top, and their mean (what a ray bouncing off
/// it sees).
#[derive(Clone, Debug, PartialEq)]
pub struct Picture {
    /// Its width and height.
    pub size: [u32; 2],
    /// Its texels, row by row.
    pub texels: Vec<[f32; 4]>,
    /// Their mean.
    pub mean: [f32; 4],
}

impl Picture {
    /// `texels` of a `size` picture, their mean taken.
    pub fn new(size: [u32; 2], texels: Vec<[f32; 4]>) -> Picture {
        let count = texels.len().max(1) as f64;
        let mean = std::array::from_fn(|k| {
            (texels.iter().map(|t| f64::from(t[k])).sum::<f64>() / count) as f32
        });
        Picture { size, texels, mean }
    }

    /// Half the size along each side longer than one, each texel the mean of
    /// those it covers.
    pub fn halved(&self) -> Picture {
        let [w, h] = self.size;
        let (sw, sh) = (u32::from(w > 1) + 1, u32::from(h > 1) + 1);
        let (nw, nh) = (w.div_ceil(sw), h.div_ceil(sh));
        let mut texels = Vec::with_capacity((nw * nh) as usize);
        for y in 0..nh {
            for x in 0..nw {
                let mut sum = [0.0f32; 4];
                let mut n = 0.0;
                for dy in 0..sh {
                    for dx in 0..sw {
                        let (sx, sy) = ((x * sw + dx).min(w - 1), (y * sh + dy).min(h - 1));
                        let t = self.texels[(sy * w + sx) as usize];
                        sum = std::array::from_fn(|k| sum[k] + t[k]);
                        n += 1.0;
                    }
                }
                texels.push(sum.map(|v| v / n));
            }
        }
        Picture::new([nw, nh], texels)
    }
}

/// How a picture's coordinates read past its edges, along one axis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Wrap {
    /// It repeats.
    #[default]
    Repeat,
    /// Its edge texel stretches on.
    Clamp,
    /// It repeats mirrored.
    Mirror,
    /// Black (MaterialX's `constant`, its default) outside.
    Black,
}

/// A surface's texture: its color times a picture, read at the transformed
/// texture coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Texture {
    /// The picture, an index into [`Scene::pictures`].
    pub picture: u32,
    /// Where corner coordinates `(u, v)` read it, by rows:
    /// `transform · (u, v, 1)`, v up (MaterialX's; the picture's rows run
    /// down).
    pub transform: [[f32; 3]; 2],
    /// How each axis reads past the edges.
    pub wrap: [Wrap; 2],
    /// Nearest texel (pixel art's `closest`), else bilinear.
    pub nearest: bool,
    /// What a texel is multiplied by (rgb and alpha).
    pub factor: [f32; 4],
    /// The alpha below which the surface is cut out (none: never).
    pub cutout: Option<f32>,
}

/// A mesh placed in the world with a material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    /// Its mesh, an index into [`Scene::meshes`].
    pub mesh: u32,
    /// Its own space to world: three rows.
    pub to_world: [[f32; 4]; 3],
    /// World to its own space: three rows (none for a flat placement, which
    /// no ray enters).
    pub to_own: Option<[[f32; 4]; 3]>,
    /// Its material, an index into [`Scene::surfaces`] (past the end: the
    /// default Pixel Surface).
    pub surface: u32,
    /// The object id it writes.
    pub id: u32,
    /// What stands for its mesh's corners' tint: its instance's own value of
    /// the attribute its material reads (none: the corners').
    pub tint: Option<[f32; 3]>,
    /// What stands for its mesh's corners' alpha, likewise: its instance's
    /// own value of the alpha attribute its material reads.
    pub alpha: Option<f32>,
    /// The slots of the geometry lights it casts no shadow of
    /// (`crate::shadowless`), 32 a word.
    pub shadowless: [u32; 4],
}

/// A Pixel Surface as a frame shades it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surface {
    /// Its albedo.
    pub color: [f32; 3],
    /// The share of light reaching its back that comes through, and that a
    /// shadow ray crossing it keeps (0: opaque).
    pub translucency: f32,
    /// What that light is tinted by.
    pub translucency_color: [f32; 3],
    /// Unlit (MaterialX's Surface Unlit): what every ray sees is its color,
    /// whatever lights it.
    pub unlit: bool,
    /// The picture its color is multiplied by, when it samples one.
    pub texture: Option<Texture>,
}

impl Default for Surface {
    /// A light grey, opaque.
    fn default() -> Surface {
        Surface {
            color: [0.8; 3],
            translucency: 0.0,
            translucency_color: [1.0; 3],
            unlit: false,
            texture: None,
        }
    }
}

impl Surface {
    /// What it lets through: translucency × translucency color.
    pub fn transmittance(&self) -> [f32; 3] {
        let t = self.translucency.clamp(0.0, 1.0);
        self.translucency_color.map(|c| c.clamp(0.0, 1.0) * t)
    }
}

/// A closed mesh filled with a medium.
#[derive(Clone, Debug, PartialEq)]
pub struct Volume {
    /// Its bounds' triangles: three corners a triangle, world space.
    pub corners: Vec<[f32; 3]>,
    /// World to the mesh's own space: three rows.
    pub to_own: [[f32; 4]; 3],
    /// Its bounding box in its own space.
    pub own_bounds: ([f32; 3], [f32; 3]),
    /// Scene units a unit of its own space spans along each axis.
    pub scale: [f32; 3],
    /// What fills it.
    pub medium: Medium,
}

/// A Pixel Volume's medium.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Medium {
    /// Extinction a unit of distance.
    pub density: f32,
    /// The single-scattering albedo.
    pub scatter_color: [f32; 3],
    /// Henyey–Greenstein g, −0.9 to 0.9.
    pub anisotropy: f32,
    /// Light given off a unit of density and distance, its color times its
    /// strength.
    pub emission: [f32; 3],
    /// How much the noise shapes the density (0: even).
    pub noise_amount: f32,
    /// Noise cycles a scene unit.
    pub noise_frequency: f32,
    /// Noise octaves, 1 to 8.
    pub noise_octaves: u32,
    /// Moves the noise, scene units.
    pub noise_offset: [f32; 3],
    /// The noise's seed.
    pub noise_seed: u32,
    /// How far in from the bounds the density fades in (0: a hard edge).
    pub edge_falloff: f32,
    /// Whether it dims the lights' light passing through it.
    pub shadows: bool,
}

/// The domes' light: their radiance from above and from below the horizon,
/// summed, those lighting through portals apart, and whether the camera
/// sees the others behind the scene.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ambient {
    /// From above.
    pub sky: [f32; 3],
    /// From below (a dome's ground color).
    pub ground: [f32; 3],
    /// From above, through portals ([`Scene::portals`]).
    pub through_sky: [f32; 3],
    /// From below, through portals.
    pub through_ground: [f32; 3],
    /// The camera sees the domes lighting without portals.
    pub seen: bool,
}

/// A portal domes light through (ambient.wgsl): its centre, half extents
/// and the way its light passes (its -Z), in the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Portal {
    /// Its centre.
    pub centre: [f32; 3],
    /// Half its width along its X.
    pub u: [f32; 3],
    /// Half its height along its Y.
    pub v: [f32; 3],
    /// The way its light passes, unit.
    pub passes: [f32; 3],
}

/// The rim lights' light, summed by where it shows.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rims {
    /// Only where direct light reaches a surface.
    pub lit: [f32; 3],
    /// All round the silhouette.
    pub all: [f32; 3],
}

/// Exponential height fog.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fog {
    /// Its color.
    pub color: [f32; 3],
    /// Its density at its height, a unit of distance.
    pub density: f32,
    /// How far from the camera it starts.
    pub start: f32,
    /// Where its density holds.
    pub height: f32,
    /// e-folds a unit up.
    pub falloff: f32,
    /// Lit by the lights (marched with the volumes) rather than flat.
    pub lit: bool,
    /// Lit fog's Henyey–Greenstein g.
    pub anisotropy: f32,
}
