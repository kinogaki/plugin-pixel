//! The scene Pixel Render draws: meshes, volumes, Pixel materials, lights,
//! an optional dome, rim lights, fog and a camera.
//!
//! The camera is `kinogaki-render-api`'s own; meshes and lights are the parts
//! of its prims Pixel Render reads (a mesh is one material's triangles,
//! a light's color its emission's), made by the scene store
//! ([`crate::store`]). Matrices are column-major `[[f32; 4]; 4]`:
//! `transform[3]` is the translation. Colors are linear ACEScg.

use std::sync::Arc;

pub use kinogaki_render_api::standard::Fog;
pub use kinogaki_render_api::{Camera, IDENTITY, Matrix, Projection};

pub use crate::gobo::Gobo;
pub use crate::rim::RimSide;
use crate::volume::PixelVolume;

/// A triangle mesh in its own space.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    /// Point positions.
    pub points: Arc<[[f32; 3]]>,
    /// Three point indices a triangle.
    pub triangles: Arc<[[u32; 3]]>,
    /// Shading normals, three a triangle (one a corner); none: flat.
    pub normals: Option<Arc<[[f32; 3]]>>,
    /// What its material's Color is multiplied by, three a triangle (one a
    /// corner, linear ACEScg): the mesh attribute its Pixel Surface reads;
    /// none: white.
    pub tints: Option<Arc<[[f32; 3]]>>,
    /// Object to world.
    pub transform: Matrix,
    /// Index into [`Scene::materials`]; one past the end takes
    /// [`PixelSurface::default`].
    pub material: u32,
    /// The object id the `Id` output writes.
    pub id: u32,
    /// Whether it is rendered.
    pub visible: bool,
}

/// What a Pixel material renders as.
#[derive(Clone, Debug, PartialEq)]
pub enum PixelMaterial {
    /// A shaded surface.
    Surface(PixelSurface),
    /// A medium filling the mesh; its surface is not drawn.
    Volume(PixelVolume),
}

/// A closed mesh filled with a medium: the volume's bounds. Its triangles
/// are only where rays enter and leave it; nothing draws or shadows them.
#[derive(Clone, Debug, PartialEq)]
pub struct Volume {
    /// The bounds (its `material` is unused).
    pub mesh: Mesh,
    /// What fills it.
    pub medium: PixelVolume,
}

/// The Pixel surface material: a surface of one color, times a mesh color
/// attribute where the mesh has one, lit and quantized; thin and
/// translucent when asked, so light behind it shows through and passes on
/// to what lies behind it.
#[derive(Clone, Debug, PartialEq)]
pub struct PixelSurface {
    /// Its albedo, linear ACEScg.
    pub color: [f32; 3],
    /// The mesh attribute that multiplies `color` (`Cd`); empty: none.
    pub attribute: Arc<str>,
    /// The share of light reaching its back face that comes through, and
    /// that a shadow ray crossing it keeps (0: opaque).
    pub translucency: f32,
    /// What the light coming through is tinted by, on top of the albedo
    /// (white: the albedo's own color), linear ACEScg.
    pub translucency_color: [f32; 3],
}

impl PixelSurface {
    /// What light reaching its back face is multiplied by before the
    /// albedo: translucency × translucency color.
    pub fn transmittance(&self) -> [f32; 3] {
        let t = self.translucency.clamp(0.0, 1.0);
        self.translucency_color.map(|c| c.max(0.0) * t)
    }
}

impl Default for PixelSurface {
    /// Light grey, opaque, reading no attribute: what renders where no
    /// material is named.
    fn default() -> PixelSurface {
        PixelSurface {
            color: [0.8; 3],
            attribute: Arc::from(""),
            translucency: 0.0,
            translucency_color: [1.0; 3],
        }
    }
}

/// A light's shape (UsdLux): area lights in the light's XY plane emitting
/// down -Z, the sun and spots.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightShape {
    /// A RectLight, `width` × `height`.
    Rect {
        /// Along X.
        width: f32,
        /// Along Y.
        height: f32,
    },
    /// A DiskLight.
    Disk {
        /// Its radius.
        radius: f32,
    },
    /// A DistantLight: parallel light down -Z from a disk `angle` degrees
    /// across (0: a delta direction).
    Distant {
        /// Its size across, degrees.
        angle: f32,
    },
    /// A SphereLight shaped into a spot (the render interface's
    /// [`kinogaki_render_api::standard::cone_falloff`]).
    Spot {
        /// Its radius.
        radius: f32,
        /// The cone's half-angle, degrees.
        cone_angle: f32,
        /// The share of the cone its edge fades over.
        cone_softness: f32,
    },
}

/// A light after UsdLux.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    /// Its shape.
    pub shape: LightShape,
    /// `inputs:color`, linear ACEScg.
    pub color: [f32; 3],
    /// `inputs:intensity`: the emitted radiance.
    pub intensity: f32,
    /// `inputs:exposure`: the intensity is scaled by 2 to this power.
    pub exposure: f32,
    /// `inputs:normalize`: its power does not change with its size.
    pub normalize: bool,
    /// Light to world.
    pub transform: Matrix,
    /// The cookie masking a Distant or Spot light ([`crate::gobo`]).
    pub gobo: Option<Gobo>,
}

impl Default for Light {
    fn default() -> Light {
        Light {
            shape: LightShape::Rect {
                width: 1.0,
                height: 1.0,
            },
            color: [1.0; 3],
            intensity: 1.0,
            exposure: 0.0,
            normalize: false,
            transform: IDENTITY,
            gobo: None,
        }
    }
}

/// The domes' light, summed (UsdLux DomeLight; an image dome lights by
/// its color): the radiance from above the horizon and from below it,
/// the same unless a dome has a ground color. It is the ambient light AO
/// scales, and the background.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Dome {
    /// Radiance from directions above the horizon, linear ACEScg.
    pub above: [f32; 3],
    /// Radiance from directions below it.
    pub below: [f32; 3],
    /// Whether the camera sees it behind the scene (else the background
    /// is transparent black).
    pub visible: bool,
}

/// The rim lights' light, summed by where it shows: each one's emission ×
/// intensity × 2^exposure.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rims {
    /// Rims shown only where direct light reaches.
    pub lit: [f32; 3],
    /// Rims shown all round.
    pub all: [f32; 3],
}

/// The scene's light: the lights but domes and rims, the domes and the rims.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Lighting {
    /// Area, distant and spot lights.
    pub lights: Vec<Light>,
    /// The domes, if any.
    pub dome: Option<Dome>,
    /// The rim lights.
    pub rims: Rims,
}

/// Everything a frame draws besides the camera.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    /// The meshes.
    pub meshes: Vec<Mesh>,
    /// The materials meshes index.
    pub materials: Vec<PixelSurface>,
    /// The volumes.
    pub volumes: Vec<Volume>,
    /// Area, distant and spot lights.
    pub lights: Vec<Light>,
    /// The domes, if any: the ambient light AO scales.
    pub dome: Option<Dome>,
    /// The rim lights.
    pub rims: Rims,
    /// The fog, if any.
    pub fog: Option<Fog>,
}

impl Scene {
    /// Its lights, domes and rims.
    pub fn lighting(&self) -> Lighting {
        Lighting {
            lights: self.lights.clone(),
            dome: self.dome,
            rims: self.rims,
        }
    }
}
