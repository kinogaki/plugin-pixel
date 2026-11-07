//! The host's prims as Pixel's scene: [`Sync`] holds every prim it was sent
//! and, as each batch arrives, says what of the [`Scene`] changed and makes
//! that again, at the shutter's opening (Pixel draws each frame sharp).
//!
//! - **Shapes and Instancers** become meshes in their own space and the
//!   instances placing them: each shape's triangles split by material into
//!   meshes, made once and kept while only the shape's placement, visibility
//!   or name changes (its dirty bits); a prim an Instancer names renders
//!   only through it, and an instance's material overrides its prototype's
//!   (meshes of their own, kept likewise). An instance's own primvars (its
//!   Instancer's, else an outer one's) give its object id (`id`, else its
//!   index + 1) and the color its material's attribute reads, which stands
//!   for its mesh's tint. Every placement of a mesh is an
//!   instance, so a move remakes the instances alone. Triangles whose
//!   material is a Pixel Volume become that volume's bounds instead, in
//!   world space, and are not drawn. A material's color attribute becomes
//!   each corner's tint (vertex, point, primitive or detail values; white
//!   where a shape lacks it); `N` its shading normals.
//! - **Materials** become surfaces and media ([`nodes::material`]); a
//!   surface's texture samples the picture its Texture prim holds (linear
//!   ACEScg, made straight, its longer side at most [`MAX_PICTURE`] texels:
//!   halved until it is), each corner reading the coordinates its primvar
//!   holds.
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
    Ambient, Corners, Fog, Instance, Light, LightShape, Mesh, Picture, Rims, Scene, Surface,
    Texture, Volume,
};
use kinogaki_render_api::geometry::image::Image;
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
    /// Each Texture prim's picture, made once.
    pictures: BTreeMap<PrimId, Arc<Picture>>,
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
            Prim::Material(_) | Prim::Texture(_) => self.materials = true,
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
        self.pictures.retain(|texture, _| !remade.contains(texture));
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

    /// The materials again, and the pictures their textures sample; whether
    /// a material's kind, tint or coordinates changed (the meshes are made
    /// again).
    fn read_materials(&mut self) -> bool {
        let before = std::mem::take(&mut self.materials);
        let mut surfaces = Vec::new();
        let mut pictures: Vec<Arc<Picture>> = Vec::new();
        let mut placed: BTreeMap<PrimId, u32> = BTreeMap::new();
        self.problems.materials.clear();
        for (id, prim) in &self.prims {
            if let Prim::Material(m) = prim {
                for node in m.network.cut() {
                    self.problems.materials.push(Diagnostic::warning(format!(
                        "{}: Pixel doesn't render `{node}`",
                        m.path
                    )));
                }
                let material = nodes::material(&m.network);
                let surface = match &material {
                    Material::Surface(s, look) => {
                        let texture = look.texture.as_ref().and_then(|t| {
                            let image = t.image?;
                            let picture = match placed.get(&image) {
                                Some(&at) => at,
                                None => {
                                    let made = match self.pictures.get(&image) {
                                        Some(made) => made.clone(),
                                        None => match self.prims.get(&image) {
                                            Some(Prim::Texture(read)) => Arc::new(picture(read)),
                                            _ => return None,
                                        },
                                    };
                                    self.pictures.insert(image, made.clone());
                                    pictures.push(made);
                                    placed.insert(image, pictures.len() as u32 - 1);
                                    pictures.len() as u32 - 1
                                }
                            };
                            Some(Texture {
                                picture,
                                transform: t.transform,
                                wrap: t.wrap,
                                nearest: t.nearest,
                                factor: t.factor,
                                cutout: look.cutout,
                            })
                        });
                        Surface { texture, ..*s }
                    }
                    Material::Volume(_) => Surface::default(),
                };
                surfaces.push(surface);
                let index = surfaces.len() as u32 - 1;
                self.materials.insert(*id, (material, index));
            }
        }
        self.pictures.retain(|id, _| placed.contains_key(id));
        self.scene.surfaces = surfaces;
        self.scene.pictures = pictures;
        let shape_of = |m: &(Material, u32)| match &m.0 {
            Material::Surface(_, look) => {
                let uv = look.texture.as_ref().map(|t| t.uv.clone());
                Some((look.tint().cloned(), uv, m.1))
            }
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
            match prim {
                _ if named.contains(id) => {}
                Prim::Shape(shape) if shape.visibility().visible => {
                    let at = shape.xform().at(0.0);
                    self.place((*id, shape), &at, None, &[], &mut placed);
                }
                Prim::Instancer(inst) if inst.visibility().visible => {
                    let at = inst.xform().at(0.0);
                    self.instances((*id, inst), &at, &mut placed, &mut Vec::new());
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

    /// Instancer `id`'s instances under `above`, below the instances of
    /// `chain`: an Instancer met again there places nothing.
    fn instances<'a>(
        &'a self,
        (id, instancer): (PrimId, &'a Instancer),
        above: &Xform,
        placed: &mut Placed,
        chain: &mut Vec<Level<'a>>,
    ) {
        for (i, xform) in instancer.xforms().iter().enumerate() {
            let at = above.compose(xform);
            let material = instancer.subsets().of(i).map(|m| m.prim());
            let prototype = instancer.prototype_of(i);
            chain.push((id, instancer, i));
            match self.prims.get(&prototype) {
                Some(Prim::Shape(shape)) => {
                    self.place((prototype, shape), &at, material, chain, placed);
                }
                Some(Prim::Instancer(_)) if chain.iter().any(|(on, ..)| *on == prototype) => {
                    placed.round = true;
                }
                Some(Prim::Instancer(nested)) => {
                    self.instances((prototype, nested), &at, placed, chain);
                }
                _ => {}
            }
            chain.pop();
        }
    }

    /// Shape `id` placed by `at`, its materials overridden by `instead`, as
    /// the instances of `chain` place it (none: on its own): an instance of
    /// each of its meshes, and its volumes in world space.
    fn place(
        &self,
        (id, shape): (PrimId, &Shape),
        at: &Xform,
        instead: Option<PrimId>,
        chain: &[Level<'_>],
        placed: &mut Placed,
    ) {
        // The innermost instance's id, else its index + 1; a shape's own
        // slot + 1.
        let object = match chain.last() {
            Some((_, instancer, i)) => (instancer.primvar(*i, names::ID))
                .and_then(|(id, at)| id.values().real(at))
                .map_or(*i as u32 + 1, |id| id as u32),
            None => id.slot + 1,
        };
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
            let tint = match self.material(*material) {
                Material::Surface(_, look) => look.tint().and_then(|name| own_color(chain, name)),
                Material::Volume(_) => None,
            };
            placed.instances.push(Instance {
                mesh: index,
                to_world,
                to_own,
                surface,
                id: object,
                tint,
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
            let group: Vec<[u32; 3]> = ts.iter().map(|&t| triangles[t]).collect();
            match self.material(material) {
                Material::Volume(_) => {
                    let corners = group.iter().flatten().map(|&p| points[p as usize]);
                    made.volumes.push((material, corners.collect()));
                }
                Material::Surface(_, look) => {
                    let tints = look.tint().and_then(|a| shape.corners::<3>(triangles, a));
                    let uvs =
                        (look.texture.as_ref()).and_then(|t| shape.corners::<2>(triangles, &t.uv));
                    let held = Corners {
                        normals: normals.as_deref().map(corners),
                        tints: tints.as_deref().map(corners),
                        uvs: uvs.as_deref().map(|list| {
                            ts.iter()
                                .flat_map(|&t| (0..3).map(move |k| list[3 * t + k]))
                                .collect()
                        }),
                    };
                    let mesh = Mesh::new(points.as_slice(), &group, held);
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

/// One of the instances placing a shape, outermost first: an Instancer's
/// prim, itself and the instance.
type Level<'a> = (PrimId, &'a Instancer, usize);

/// The color the instances of `chain` hold of primvar `name` (the innermost
/// holding one): its first three numbers, a number spread to three.
fn own_color(chain: &[Level<'_>], name: &str) -> Option<[f32; 3]> {
    let (held, at) =
        (chain.iter().rev()).find_map(|(_, instancer, i)| instancer.primvar(*i, name))?;
    let width = usize::from(held.ty().width).max(1);
    let value = |k: usize| held.values().real(at * width + k.min(width - 1));
    Some([value(0)?, value(1)?, value(2)?].map(|v| v as f32))
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

/// The longest side, in texels, of a picture a surface samples: a larger
/// one is halved until it fits (Pixel's films are coarse).
pub const MAX_PICTURE: u32 = 2048;

/// `image`'s main layer as a picture: its display window's texels, rows from
/// the top, straight where the host premultiplied them; a grey one grey in
/// every channel.
fn picture(image: &Image) -> Picture {
    let window = image.display();
    let [w, h] = window.size();
    let [x0, y0] = window.min();
    let channels = (image.layers().first()).map_or(4, |l| l.format().channels.len());
    let mut texels = Vec::with_capacity(w as usize * h as usize);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let [r, g, b, a] = image.read_pixel(0, x0 + x, y0 + y);
            let rgba = match channels {
                1 => [r, r, r, 1.0],
                2 => [r, r, r, g],
                3 => [r, g, b, 1.0],
                _ => [r, g, b, a],
            };
            texels.push(straight(rgba));
        }
    }
    let mut picture = Picture::new([w, h], texels);
    while picture.size[0].max(picture.size[1]) > MAX_PICTURE {
        picture = picture.halved();
    }
    picture
}

/// `rgba` unpremultiplied where its alpha is between 0 and 1.
fn straight([r, g, b, a]: [f32; 4]) -> [f32; 4] {
    match a > 0.0 && a < 1.0 {
        true => [r / a, g / a, b / a, a],
        false => [r, g, b, a],
    }
}
