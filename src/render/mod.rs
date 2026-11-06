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
pub mod settings;
pub mod vector;

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
    /// The closed meshes filled with a medium (never drawn).
    pub volumes: Vec<Volume>,
    /// The lights with a place or a direction.
    pub lights: Vec<Light>,
    /// The domes' light.
    pub ambient: Ambient,
    /// The rim lights' light.
    pub rims: Rims,
    /// The height fog, when there is one.
    pub fog: Option<Fog>,
}

/// Triangles of one material: three corners a triangle, in their own space,
/// with the hierarchy rays trace them by.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    /// Corner positions.
    pub corners: Vec<[f32; 3]>,
    /// Corner shading normals (none: flat).
    pub normals: Option<Vec<[f32; 3]>>,
    /// What each corner's albedo is multiplied by (the material's color
    /// attribute; none: white).
    pub tints: Option<Vec<[f32; 3]>>,
    /// The hierarchy over its triangles.
    pub bvh: bvh::Bvh,
}

impl Mesh {
    /// The mesh of `corners` (whole triangles: a partial one is dropped),
    /// its normals and tints kept only when they match them, its hierarchy
    /// built.
    pub fn new(
        mut corners: Vec<[f32; 3]>,
        normals: Option<Vec<[f32; 3]>>,
        tints: Option<Vec<[f32; 3]>>,
    ) -> Mesh {
        corners.truncate(corners.len() / 3 * 3);
        let fits = |c: Option<Vec<[f32; 3]>>| c.filter(|c| c.len() == corners.len());
        let (normals, tints) = (fits(normals), fits(tints));
        let bvh = bvh::Bvh::build(corners.as_chunks::<3>().0);
        Mesh {
            corners,
            normals,
            tints,
            bvh,
        }
    }
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
}

impl Default for Surface {
    /// A light grey, opaque.
    fn default() -> Surface {
        Surface {
            color: [0.8; 3],
            translucency: 0.0,
            translucency_color: [1.0; 3],
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

/// A light with a place or a direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    /// Its shape.
    pub shape: LightShape,
    /// Its radiance: the emission's color × intensity × 2^exposure.
    pub radiance: [f32; 3],
    /// Its power does not change with its size.
    pub normalize: bool,
    /// Light to world: three rows (it shines down its -Z).
    pub rows: [[f32; 4]; 3],
    /// Its cookie, on distant lights and spots.
    pub gobo: Option<Gobo>,
}

/// A light's shape, as UsdLux shapes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightShape {
    /// A rectangle or disk of `area` in its XY plane.
    Area {
        /// Its area in its own units.
        area: f32,
    },
    /// Parallel light `angle` degrees across.
    Distant {
        /// The sun's size across.
        angle: f32,
    },
    /// A sphere of `radius` in a cone about its -Z.
    Spot {
        /// Its radius.
        radius: f32,
        /// The cone's half-angle, degrees.
        cone_angle: f32,
        /// The share of the cone its light fades over.
        cone_softness: f32,
    },
}

/// A mask of dappled light on a distant light or spot, as if through leaves
/// out of frame: value-noise fBm on the light's plane, `scale` scene units a
/// feature, its edges `contrast` hard.
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

/// The domes' light: their radiance from above and from below the horizon,
/// summed, and whether the camera sees them behind the scene.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ambient {
    /// From above.
    pub sky: [f32; 3],
    /// From below (a dome's ground color).
    pub ground: [f32; 3],
    /// The camera sees them.
    pub seen: bool,
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
