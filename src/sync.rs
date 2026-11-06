//! The host's prims as Pixel's scene: [`Sync`] holds every prim it was sent
//! and, as each batch arrives, says what of the [`Scene`] changed and makes
//! it again, in world space, at the shutter's opening (Pixel draws each frame
//! sharp).
//!
//! - **Shapes and Instancers** become meshes: each shape's triangles split
//!   by material, every instance of a prototype flattened to world space (a
//!   prim an Instancer names renders only through it; an instance's material
//!   overrides its prototype's). Triangles whose material is a Pixel Volume
//!   become that volume's bounds instead, and are not drawn. A material's
//!   color attribute becomes each corner's tint (vertex, point, primitive or
//!   detail values; white where a shape lacks it); `N` its shading normals.
//! - **Materials** become surfaces and media ([`nodes::material`]).
//! - **Declared prims** become lights: the domes add up into the ambient
//!   light (a ground color below the horizon), the rims by their side, the
//!   first visible fog is the fog.
//!
//! Only meshes rebuild the BVH. A material's new colors rewrite the object
//! table; a new kind of material, or a new color attribute, makes the meshes
//! again; lights and cameras never touch geometry.
//!
//! What it can't render as sent it says ([`Sync::problems`]): volumes past
//! [`MAX_VOLUMES`], an Instancer placing itself (a cycle a later change can
//! make: it places nothing there), an illuminant Pixel's table lacks, a
//! material node it doesn't declare.
use crate::gpu::Changed;
use crate::gpu::MAX_VOLUMES;
use crate::lights::{self, RIM_LIGHT};
use crate::nodes::{self, Material};
use crate::render::{Ambient, Fog, Light, LightShape, Mesh, Rims, Scene, Surface, Volume};
use kinogaki_render_api::geometry::{
    Attribute, Diagnostic, Domain, UpAxis, Values, Vec3, Xform, names,
};
use kinogaki_render_api::standard::{self, Shape as LightKind};
use kinogaki_render_api::{
    Declared, Instancer, Prim, PrimId, SceneSettings, SceneUpdate, Shape, Topology, ViewCamera,
};
use std::collections::{BTreeMap, BTreeSet};

/// Every prim held, and the scene they make.
#[derive(Default)]
pub struct Sync {
    prims: BTreeMap<PrimId, Prim>,
    /// Each Material prim's material and its index in the scene's surfaces.
    materials: BTreeMap<PrimId, (Material, u32)>,
    /// The Material prim each of the scene's volumes is filled by.
    filled_by: Vec<Option<PrimId>>,
    settings: SceneSettings,
    scene: Scene,
    /// What it can't render as sent, by what found it.
    problems: Problems,
}

/// What each part of the scene can't render as sent.
#[derive(Default)]
struct Problems {
    geometry: Vec<Diagnostic>,
    materials: Vec<Diagnostic>,
}

/// Three rows of `x` in single precision.
fn rows(x: &Xform) -> [[f32; 4]; 3] {
    x.rows().map(|r| r.map(|v| v as f32))
}

/// Item `i` of `attribute` as a color: its first three numbers (one spread
/// to grey); none for text or arrays.
fn rgb(attribute: &Attribute, i: usize) -> Option<[f32; 3]> {
    if attribute.offsets().is_some() {
        return None;
    }
    let width = attribute.values().len() / attribute.items().max(1);
    let at = |k: usize| i * width + k.min(width - 1);
    let read = |k: usize| -> Option<f32> {
        Some(match attribute.values() {
            Values::F32(c) => *c.get(at(k))?,
            Values::F64(c) => *c.get(at(k))? as f32,
            Values::I32(c) => *c.get(at(k))? as f32,
            Values::I64(c) => *c.get(at(k))? as f32,
            _ => return None,
        })
    };
    Some([read(0)?, read(1)?, read(2)?])
}

/// `shape`'s primvar `name` at each triangle corner, in triangle order.
fn per_corner(shape: &Shape, triangles: &[[u32; 3]], name: &str) -> Option<Vec<[f32; 3]>> {
    let (domain, attribute) = shape.primvar(name)?;
    let corner = |t: usize, k: usize| match domain {
        Domain::Vertex => rgb(attribute, 3 * t + k),
        Domain::Point => rgb(attribute, triangles[t][k] as usize),
        Domain::Primitive => rgb(attribute, t),
        _ => rgb(attribute, 0),
    };
    (0..triangles.len())
        .flat_map(|t| (0..3).map(move |k| (t, k)))
        .map(|(t, k)| corner(t, k))
        .collect()
}

/// What a batch touched.
#[derive(Default)]
struct Touched {
    shapes: bool,
    materials: bool,
    declared: bool,
}

impl Touched {
    fn by(&mut self, prim: &Prim) {
        match prim {
            Prim::Shape(_) | Prim::Instancer(_) => self.shapes = true,
            Prim::Material(_) => self.materials = true,
            Prim::Declared(_) => self.declared = true,
            _ => {}
        }
    }
}

impl Sync {
    /// The scene the prims make.
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// What it can't render as sent, each said once.
    pub fn problems(&self) -> Vec<Diagnostic> {
        let p = &self.problems;
        [&p.geometry, &p.materials]
            .into_iter()
            .flatten()
            .cloned()
            .collect()
    }

    /// The world's up direction.
    pub fn up(&self) -> [f32; 3] {
        match self.settings.units.up() {
            UpAxis::Z => [0.0, 0.0, 1.0],
            _ => [0.0, 1.0, 0.0],
        }
    }

    /// Applies `update`, saying what of the scene changed.
    pub fn apply(&mut self, update: SceneUpdate) -> Changed {
        let mut touched = Touched::default();
        for id in &update.removed {
            if let Some(prim) = self.prims.remove(id) {
                touched.by(&prim);
            }
        }
        for (id, prim) in update.added {
            touched.by(&prim);
            self.prims.insert(id, prim);
        }
        for change in update.changed {
            touched.by(&change.prim);
            self.prims.insert(change.id, change.prim);
        }
        if let Some(settings) = update.settings {
            self.settings = settings;
            touched.declared = true;
        }
        let mut changed = Changed::default();
        if touched.materials {
            let reshaped = self.read_materials();
            changed.surfaces = true;
            changed.volumes = true;
            touched.shapes |= reshaped;
        }
        if touched.shapes {
            self.build_geometry();
            changed.geometry = true;
            changed.volumes = true;
        } else if touched.materials {
            self.restyle_volumes();
        }
        if touched.declared {
            self.read_lights();
            changed.lights = true;
        }
        changed
    }

    /// The materials again; whether a material's kind or color attribute
    /// changed (the meshes are made again).
    fn read_materials(&mut self) -> bool {
        let before = std::mem::take(&mut self.materials);
        let mut surfaces = Vec::new();
        self.problems.materials.clear();
        for (id, prim) in &self.prims {
            if let Prim::Material(m) = prim {
                if let Some(def) = nodes::foreign(&m.network) {
                    self.problems.materials.push(Diagnostic::warning(format!(
                        "{def} isn't a Pixel material: it renders the default surface"
                    )));
                }
                let material = nodes::material(&m.network);
                if let Material::Surface(s, _) = &material {
                    surfaces.push(*s);
                } else {
                    surfaces.push(Surface::default());
                }
                let index = surfaces.len() as u32 - 1;
                self.materials.insert(*id, (material, index));
            }
        }
        self.scene.surfaces = surfaces;
        let shape_of = |m: &(Material, u32)| match &m.0 {
            Material::Surface(_, attribute) => Some((attribute.clone(), m.1)),
            Material::Volume(_) => None,
        };
        let same = before.len() == self.materials.len()
            && (before.iter().zip(&self.materials))
                .all(|((a, x), (b, y))| a == b && shape_of(x) == shape_of(y));
        !same
    }

    /// The material `id` names: its index, or none for a volume (and the
    /// default for none).
    fn material(&self, id: Option<PrimId>) -> &Material {
        static DEFAULT: std::sync::LazyLock<Material> = std::sync::LazyLock::new(Material::default);
        id.and_then(|id| self.materials.get(&id))
            .map_or(&DEFAULT, |m| &m.0)
    }

    /// Every top-level shape and instancer, made again.
    fn build_geometry(&mut self) {
        let named: BTreeSet<PrimId> = (self.prims.values())
            .filter_map(|p| match p {
                Prim::Instancer(i) => Some(i.prototypes().iter().copied()),
                _ => None,
            })
            .flatten()
            .collect();
        let mut built = Built::default();
        for (id, prim) in &self.prims {
            let object = id.slot + 1;
            match prim {
                _ if named.contains(id) => {}
                Prim::Shape(shape) if shape.visibility().visible => {
                    self.place(shape, &shape.xform().at(0.0), None, object, &mut built);
                }
                Prim::Instancer(inst) if inst.visibility().visible => {
                    let at = inst.xform().at(0.0);
                    self.instances(inst, &at, object, &mut built, &mut vec![*id]);
                }
                _ => {}
            }
        }
        self.scene.meshes = built.meshes;
        self.problems.geometry.clear();
        if built.round {
            self.problems.geometry.push(Diagnostic::warning(
                "an Instancer places itself: its cycle places nothing",
            ));
        }
        if built.volumes.len() > MAX_VOLUMES {
            self.problems.geometry.push(Diagnostic::warning(format!(
                "{} volumes: Pixel renders the first {MAX_VOLUMES}",
                built.volumes.len()
            )));
        }
        (self.scene.volumes, self.filled_by) = built.volumes.into_iter().unzip();
    }

    /// `instancer`'s instances under `above`, it and the Instancers above it
    /// on `path`: one met again places nothing there.
    fn instances(
        &self,
        instancer: &Instancer,
        above: &Xform,
        object: u32,
        built: &mut Built,
        path: &mut Vec<PrimId>,
    ) {
        for (i, xform) in instancer.xforms().iter().enumerate() {
            let at = above.compose(xform);
            let material = instancer.subsets().of(i).map(|m| m.prim());
            let prototype = instancer.prototype_of(i);
            match self.prims.get(&prototype) {
                Some(Prim::Shape(shape)) => self.place(shape, &at, material, object, built),
                Some(Prim::Instancer(_)) if path.contains(&prototype) => built.round = true,
                Some(Prim::Instancer(nested)) => {
                    path.push(prototype);
                    self.instances(nested, &at, object, built, path);
                    path.pop();
                }
                _ => {}
            }
        }
    }

    /// `shape` placed by `at`, its materials overridden by `instead`.
    fn place(
        &self,
        shape: &Shape,
        at: &Xform,
        instead: Option<PrimId>,
        object: u32,
        built: &mut Built,
    ) {
        let Topology::Triangles(triangles) = shape.topology() else {
            return;
        };
        let triangles: &[[u32; 3]] = triangles.as_slice();
        let world: Vec<[f32; 3]> = (shape.points().iter())
            .map(|p| at.apply_point(Vec3::from(p.map(f64::from))).to_f32())
            .collect();
        let normals = per_corner(shape, triangles, names::N).map(|n| {
            let inverse = at.inverse().unwrap_or(Xform::IDENTITY);
            let r = inverse.rows();
            let turned = |v: [f32; 3]| -> [f32; 3] {
                std::array::from_fn(|i| {
                    (0..3).map(|j| r[j][i] * f64::from(v[j])).sum::<f64>() as f32
                })
            };
            n.into_iter().map(turned).collect::<Vec<_>>()
        });
        // Triangles by the material they render with.
        let mut groups: BTreeMap<Option<PrimId>, Vec<usize>> = BTreeMap::new();
        for t in 0..triangles.len() {
            let material = instead.or_else(|| shape.subsets().of(t).map(|m| m.prim()));
            groups.entry(material).or_default().push(t);
        }
        for (material_id, ts) in groups {
            let corners = |list: &[[f32; 3]]| -> Vec<[f32; 3]> {
                ts.iter()
                    .flat_map(|&t| (0..3).map(move |k| list[3 * t + k]))
                    .collect()
            };
            let positions: Vec<[f32; 3]> = ts
                .iter()
                .flat_map(|&t| triangles[t].map(|p| world[p as usize]))
                .collect();
            match self.material(material_id) {
                Material::Volume(medium) => {
                    built
                        .volumes
                        .push((volume(shape, at, positions, *medium), material_id));
                }
                Material::Surface(_, attribute) => {
                    let tints = attribute
                        .as_ref()
                        .and_then(|a| per_corner(shape, triangles, a));
                    let surface = material_id
                        .and_then(|m| self.materials.get(&m))
                        .map_or(u32::MAX, |m| m.1);
                    built.meshes.push(Mesh {
                        corners: positions,
                        normals: normals.as_deref().map(corners),
                        tints: tints.as_deref().map(corners),
                        surface,
                        id: object,
                    });
                }
            }
        }
    }

    /// The volumes' media again, their bounds as they were.
    fn restyle_volumes(&mut self) {
        let media: Vec<Option<crate::render::Medium>> = (self.filled_by.iter())
            .map(|id| match self.material(*id) {
                Material::Volume(medium) => Some(*medium),
                Material::Surface(..) => None,
            })
            .collect();
        for (volume, medium) in self.scene.volumes.iter_mut().zip(media) {
            if let Some(medium) = medium {
                volume.medium = medium;
            }
        }
    }

    /// The lights, domes, rims and fog again.
    fn read_lights(&mut self) {
        let (mut lights, mut ambient, mut rims, mut fog) =
            (Vec::new(), Ambient::default(), Rims::default(), None);
        for prim in self.prims.values() {
            let Prim::Declared(prim) = prim else {
                continue;
            };
            if !prim.visibility.visible {
                continue;
            }
            if &*prim.type_id == standard::FOG {
                fog = fog.or_else(|| Some(fog_of(prim)));
                continue;
            }
            if &*prim.type_id == RIM_LIGHT {
                let side = if lights::rim_all_round(prim) {
                    &mut rims.all
                } else {
                    &mut rims.lit
                };
                add(side, lights::emitted(prim));
                continue;
            }
            let Some(light) = standard::Light::decode(prim) else {
                continue;
            };
            let radiance = light.radiance();
            let shape = match light.shape {
                LightKind::Dome { ground, .. } => {
                    add(&mut ambient.sky, radiance);
                    let scale = light.intensity * light.exposure.exp2();
                    add(
                        &mut ambient.ground,
                        ground.map_or(radiance, |g| g.map(|c| c * scale)),
                    );
                    ambient.seen |= prim.visibility.camera;
                    continue;
                }
                LightKind::Rect { width, height } => LightShape::Area {
                    area: width * height,
                },
                LightKind::Disk { radius } => LightShape::Area {
                    area: std::f32::consts::PI * radius * radius,
                },
                LightKind::Distant { angle } => LightShape::Distant { angle },
                LightKind::Spot {
                    radius,
                    cone_angle,
                    cone_softness,
                } => LightShape::Spot {
                    radius,
                    cone_angle,
                    cone_softness,
                },
            };
            lights.push(Light {
                shape,
                radiance,
                normalize: light.normalize,
                rows: rows(&prim.xform.at(0.0)),
                gobo: lights::gobo(prim),
            });
        }
        (
            self.scene.lights,
            self.scene.ambient,
            self.scene.rims,
            self.scene.fog,
        ) = (lights, ambient, rims, fog);
    }

    /// The camera `camera` names, as the renderer takes it (none: a prim it
    /// doesn't hold).
    pub fn camera(&self, camera: &ViewCamera) -> Option<crate::render::camera::Camera> {
        let camera = match camera {
            ViewCamera::Free(c) => c,
            ViewCamera::Prim(id) => match self.prims.get(&id.prim()) {
                Some(Prim::Camera(c)) => c,
                _ => return None,
            },
        };
        Some(crate::render::camera::Camera {
            orthographic: camera.projection == kinogaki_render_api::Projection::Orthographic,
            focal_length: camera.focal_length,
            aperture: camera.horizontal_aperture,
            offset: [
                camera.horizontal_aperture_offset,
                camera.vertical_aperture_offset,
            ],
            clip: camera.clipping_range,
            mm_per_unit: self.settings.units.millimeters_per_unit() as f32,
            rows: *camera.xform.at(0.0).rows(),
        })
    }
}

/// What the shapes make.
#[derive(Default)]
struct Built {
    meshes: Vec<Mesh>,
    volumes: Vec<(Volume, Option<PrimId>)>,
    /// An Instancer placed itself.
    round: bool,
}

/// Adds `b` into `a`.
fn add(a: &mut [f32; 3], b: [f32; 3]) {
    *a = std::array::from_fn(|k| a[k] + b[k]);
}

/// A volume of `shape`'s triangles `corners` (world space) placed by `at`.
fn volume(
    shape: &Shape,
    at: &Xform,
    corners: Vec<[f32; 3]>,
    medium: crate::render::Medium,
) -> Volume {
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for p in shape.points().iter() {
        for k in 0..3 {
            (lo[k], hi[k]) = (lo[k].min(p[k]), hi[k].max(p[k]));
        }
    }
    let to_own = at.inverse().unwrap_or(Xform::IDENTITY);
    Volume {
        corners,
        to_own: rows(&to_own),
        own_bounds: (lo, hi),
        scale: at.axes().map(|a| a.length() as f32),
        medium,
    }
}

/// The fog `prim` describes.
fn fog_of(prim: &Declared) -> Fog {
    Fog {
        color: prim.color("color"),
        density: prim.float("density").max(0.0),
        start: prim.float("start").max(0.0),
        height: prim.float("height"),
        falloff: prim.float("falloff").max(0.0),
        lit: prim.bool("inscatter"),
        anisotropy: prim.float("anisotropy").clamp(-0.9, 0.9),
    }
}
