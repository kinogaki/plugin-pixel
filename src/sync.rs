//! The host's prims as Pixel's scene: [`Sync`] holds every prim it was sent
//! and, as each batch arrives, says what of the [`Scene`] changed and makes
//! that again, at the shutter's opening (Pixel draws each frame sharp).
//!
//! - **Shapes and Instancers** become meshes in their own space and the
//!   instances placing them: each shape's triangles split by material into
//!   meshes, made once and kept while only the shape's placement, visibility
//!   or name changes (its dirty bits); a prim an Instancer names renders
//!   only through it, and an instance's material overrides its prototype's
//!   (meshes of their own, kept likewise). Every placement of a mesh is an
//!   instance, so a move remakes the instances alone. Triangles whose
//!   material is a Pixel Volume become that volume's bounds instead, in
//!   world space, and are not drawn. A material's color attribute becomes
//!   each corner's tint (vertex, point, primitive or detail values; white
//!   where a shape lacks it); `N` its shading normals.
//! - **Materials** become surfaces and media ([`nodes::material`]).
//! - **Declared prims** become lights: the domes add up into the ambient
//!   light (a ground color below the horizon), the rims by their side, the
//!   first visible fog is the fog.
//!
//! A material's new colors rewrite the instances' surfaces alone; a new
//! kind of material, or a new color attribute, makes the meshes again;
//! lights and cameras never touch geometry.
//!
//! What it can't render as sent it says ([`Sync::problems`]): volumes past
//! [`MAX_VOLUMES`], an Instancer placing itself (a cycle a later change can
//! make: it places nothing there), a material node it doesn't declare.
use crate::gpu::Changed;
use crate::gpu::MAX_VOLUMES;
use crate::lights::{self, RIM_LIGHT};
use crate::nodes::{self, Material};
use crate::render::{
    Ambient, Fog, Instance, Light, LightShape, Mesh, Rims, Scene, Surface, Volume,
};
use kinogaki_render_api::geometry::{Diagnostic, Vec3, Xform, names};
use kinogaki_render_api::standard::{self, Shape as LightKind};
use kinogaki_render_api::{
    Dirty, Instancer, Prim, PrimId, SceneSettings, SceneUpdate, Shape, Topology, ViewCamera,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Every prim held, and the scene they make.
#[derive(Default)]
pub struct Sync {
    prims: BTreeMap<PrimId, Prim>,
    /// Each Material prim's material and its index in the scene's surfaces.
    materials: BTreeMap<PrimId, (Material, u32)>,
    /// The Material prim each of the scene's volumes is filled by.
    filled_by: Vec<Option<PrimId>>,
    /// What each placed shape makes, by the shape and the material an
    /// instance overrides its own with.
    made: BTreeMap<(PrimId, Option<PrimId>), Arc<Made>>,
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

/// What a shape makes under one choice of materials, in its own space.
#[derive(Default)]
struct Made {
    /// Its surface triangles by material.
    meshes: Vec<(Option<PrimId>, Arc<Mesh>)>,
    /// Its volume triangles by material: three corners a triangle.
    volumes: Vec<(Option<PrimId>, Vec<[f32; 3]>)>,
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

/// What changes a shape's meshes: a placement, visibility or name doesn't.
const RESHAPED: Dirty = Dirty::POINTS
    .union(Dirty::TOPOLOGY)
    .union(Dirty::PRIMVARS)
    .union(Dirty::MATERIALS);

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
        self.settings.units.up().direction().to_f32()
    }

    /// Applies `update`, saying what of the scene changed.
    pub fn apply(&mut self, update: SceneUpdate) -> Changed {
        let mut touched = Touched::default();
        let mut remade = BTreeSet::new();
        for id in &update.removed {
            if let Some(prim) = self.prims.remove(id) {
                touched.by(&prim);
                remade.insert(*id);
            }
        }
        for (id, prim) in update.added {
            touched.by(&prim);
            remade.insert(id);
            self.prims.insert(id, prim);
        }
        for change in update.changed {
            touched.by(&change.prim);
            if change.dirty.intersects(RESHAPED) {
                remade.insert(change.id);
            }
            self.prims.insert(change.id, change.prim);
        }
        self.made.retain(|(shape, _), _| !remade.contains(shape));
        if let Some(settings) = update.settings {
            self.settings = settings;
            touched.declared = true;
        }
        let mut changed = Changed::default();
        if touched.materials {
            if self.read_materials() {
                self.made.clear();
                touched.shapes = true;
            }
            changed.surfaces = true;
        }
        if touched.shapes {
            let (meshes, volumes) = (self.scene.meshes.clone(), self.scene.volumes.clone());
            self.place_all();
            changed.meshes = !same_meshes(&meshes, &self.scene.meshes);
            changed.instances = true;
            changed.volumes = volumes != self.scene.volumes;
        } else if touched.materials {
            let volumes = self.scene.volumes.clone();
            self.restyle_volumes();
            changed.volumes = volumes != self.scene.volumes;
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

    /// Every top-level shape and Instancer placed again: the instances and
    /// volumes, the meshes as kept or made.
    fn place_all(&mut self) {
        let named: BTreeSet<PrimId> = (self.prims.values())
            .filter_map(|p| match p {
                Prim::Instancer(i) => Some(i.prototypes().iter().copied()),
                _ => None,
            })
            .flatten()
            .collect();
        let mut placed = Placed {
            kept: std::mem::take(&mut self.made),
            ..Placed::default()
        };
        for (id, prim) in &self.prims {
            let object = id.slot + 1;
            match prim {
                _ if named.contains(id) => {}
                Prim::Shape(shape) if shape.visibility().visible => {
                    let at = shape.xform().at(0.0);
                    self.place((*id, shape), &at, None, object, &mut placed);
                }
                Prim::Instancer(inst) if inst.visibility().visible => {
                    let at = inst.xform().at(0.0);
                    self.instances(inst, &at, object, &mut placed, &mut vec![*id]);
                }
                _ => {}
            }
        }
        self.made = placed.made;
        self.scene.meshes = placed.meshes;
        self.scene.instances = placed.instances;
        self.problems.geometry.clear();
        if placed.round {
            self.problems.geometry.push(Diagnostic::warning(
                "an Instancer places itself: its cycle places nothing",
            ));
        }
        if placed.volumes.len() > MAX_VOLUMES {
            self.problems.geometry.push(Diagnostic::warning(format!(
                "{} volumes: Pixel renders the first {MAX_VOLUMES}",
                placed.volumes.len()
            )));
        }
        (self.scene.volumes, self.filled_by) = placed.volumes.into_iter().unzip();
    }

    /// `instancer`'s instances under `above`, it and the Instancers above it
    /// on `path`: one met again places nothing there.
    fn instances(
        &self,
        instancer: &Instancer,
        above: &Xform,
        object: u32,
        placed: &mut Placed,
        path: &mut Vec<PrimId>,
    ) {
        for (i, xform) in instancer.xforms().iter().enumerate() {
            let at = above.compose(xform);
            let material = instancer.subsets().of(i).map(|m| m.prim());
            let prototype = instancer.prototype_of(i);
            match self.prims.get(&prototype) {
                Some(Prim::Shape(shape)) => {
                    self.place((prototype, shape), &at, material, object, placed);
                }
                Some(Prim::Instancer(_)) if path.contains(&prototype) => placed.round = true,
                Some(Prim::Instancer(nested)) => {
                    path.push(prototype);
                    self.instances(nested, &at, object, placed, path);
                    path.pop();
                }
                _ => {}
            }
        }
    }

    /// Shape `id` placed by `at`, its materials overridden by `instead`: an
    /// instance of each of its meshes, and its volumes in world space.
    fn place(
        &self,
        (id, shape): (PrimId, &Shape),
        at: &Xform,
        instead: Option<PrimId>,
        object: u32,
        placed: &mut Placed,
    ) {
        let key = (id, instead);
        let made = match placed.made.get(&key).or_else(|| placed.kept.get(&key)) {
            Some(made) => made.clone(),
            None => Arc::new(self.make(shape, instead)),
        };
        let (to_world, to_own) = (rows(at), at.inverse().map(|back| rows(&back)));
        for (material, mesh) in &made.meshes {
            let index = *placed.index.entry(Arc::as_ptr(mesh)).or_insert_with(|| {
                placed.meshes.push(mesh.clone());
                placed.meshes.len() as u32 - 1
            });
            let surface = material
                .and_then(|m| self.materials.get(&m))
                .map_or(u32::MAX, |m| m.1);
            placed.instances.push(Instance {
                mesh: index,
                to_world,
                to_own,
                surface,
                id: object,
            });
        }
        for (material, corners) in &made.volumes {
            if let Material::Volume(medium) = self.material(*material) {
                let world = (corners.iter())
                    .map(|p| at.apply_point(Vec3::from(p.map(f64::from))).to_f32())
                    .collect();
                placed
                    .volumes
                    .push((volume(shape, at, world, *medium), *material));
            }
        }
        placed.made.insert(key, made);
    }

    /// What `shape` makes with its materials overridden by `instead`, in its
    /// own space: its triangles by material, meshes for surfaces and corners
    /// for volumes.
    fn make(&self, shape: &Shape, instead: Option<PrimId>) -> Made {
        let mut made = Made::default();
        let Topology::Triangles(triangles) = shape.topology() else {
            return made;
        };
        let triangles: &[[u32; 3]] = triangles.as_slice();
        let points = shape.points();
        let normals = shape.corners::<3>(triangles, names::N);
        // Triangles by the material they render with.
        let mut groups: BTreeMap<Option<PrimId>, Vec<usize>> = BTreeMap::new();
        for t in 0..triangles.len() {
            let material = instead.or_else(|| shape.subsets().of(t).map(|m| m.prim()));
            groups.entry(material).or_default().push(t);
        }
        for (material, ts) in groups {
            let corners = |list: &[[f32; 3]]| -> Vec<[f32; 3]> {
                ts.iter()
                    .flat_map(|&t| (0..3).map(move |k| list[3 * t + k]))
                    .collect()
            };
            let positions: Vec<[f32; 3]> = ts
                .iter()
                .flat_map(|&t| triangles[t].map(|p| points[p as usize]))
                .collect();
            match self.material(material) {
                Material::Volume(_) => made.volumes.push((material, positions)),
                Material::Surface(_, attribute) => {
                    let tints = attribute
                        .as_ref()
                        .and_then(|a| shape.corners::<3>(triangles, a));
                    let mesh = Mesh::new(
                        positions,
                        normals.as_deref().map(corners),
                        tints.as_deref().map(corners),
                    );
                    made.meshes.push((material, Arc::new(mesh)));
                }
            }
        }
        made
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
            if let Some(standard) = standard::Fog::decode(prim) {
                fog = fog.or_else(|| Some(fog_of(standard)));
                continue;
            }
            if prim.type_id == RIM_LIGHT {
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

/// What placing the shapes makes.
#[derive(Default)]
struct Placed {
    /// What the shapes made before, taken as they are placed again.
    kept: BTreeMap<(PrimId, Option<PrimId>), Arc<Made>>,
    /// What the placed shapes make.
    made: BTreeMap<(PrimId, Option<PrimId>), Arc<Made>>,
    meshes: Vec<Arc<Mesh>>,
    /// Each mesh's index in `meshes`.
    index: BTreeMap<*const Mesh, u32>,
    instances: Vec<Instance>,
    volumes: Vec<(Volume, Option<PrimId>)>,
    /// An Instancer placed itself.
    round: bool,
}

/// Whether `a` and `b` are the same meshes, in the same order.
fn same_meshes(a: &[Arc<Mesh>], b: &[Arc<Mesh>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| Arc::ptr_eq(a, b))
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

/// The standard fog `fog` as Pixel renders it.
fn fog_of(fog: standard::Fog) -> Fog {
    Fog {
        color: fog.color,
        density: fog.density,
        start: fog.start,
        height: fog.height,
        falloff: fog.falloff,
        lit: fog.inscatter,
        anisotropy: fog.anisotropy,
    }
}
