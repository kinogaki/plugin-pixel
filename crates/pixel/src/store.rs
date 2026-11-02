//! The scene a host's prims make: [`Store`] keeps the prims it is sent
//! (`kinogaki_render_api::SceneUpdate`s) and gives Pixel Render's scene from
//! them, saying what each update changed so a renderer redoes only that.
//!
//! - A mesh's or an instancer's points, triangles, normals, material slots,
//!   attributes, placement or visibility change the **geometry** (the BVH
//!   is rebuilt), as does a material naming another color attribute.
//! - A material changes the **materials** (the object table), a light the
//!   **lights**, a camera the **cameras** and the settings the
//!   **settings**: none of them touch the geometry.
//!
//! A mesh becomes one Pixel mesh a material its triangles name; materials
//! are indexed by their prim's slot, so adding or removing one never
//! renumbers another. The triangles naming a Pixel Volume become a volume
//! (its bounds) instead of a mesh, so a material turning from surface to
//! volume, or back, changes the geometry; a volume's inputs change only the
//! materials. Each Pixel mesh carries the color attribute its
//! material reads as corner tints ([`crate::tint`]). Instancers draw copies
//! of their prototype at their placements, besides the prototype's own.
//! Emitters show their emission's color (a blackbody's or an illuminant's
//! at luminance 1). Lights and fog come as declared prims, decoded by
//! their types: the standard lights (distant lights and spots with their
//! gobos), Pixel's rim light and the standard fog. Domes add up into the
//! ambient light (a ground color below the horizon), rim lights into the
//! rim; the first fog prim is the scene's fog (hosts send one).

use std::collections::BTreeMap;
use std::sync::Arc;

use kinogaki_render_api::standard::{self, Emission, Fog, Shape};
use kinogaki_render_api::{
    Declared, Dirty, Instancer, Mesh as RenderMesh, Prim, PrimId, RenderSettings, SceneUpdate,
};

use crate::gobo::Gobo;
use crate::rim::{Rim, RimSide};
use crate::scene::{
    Camera, Dome, Light, LightShape, Lighting, Matrix, Mesh, PixelMaterial, PixelSurface, Volume,
};

/// What an update changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Changed {
    /// Meshes or instancers: the geometry and its BVH.
    pub geometry: bool,
    /// Materials: the object table and the volumes' media.
    pub materials: bool,
    /// Lights, domes or rims.
    pub lights: bool,
    /// The fog.
    pub fog: bool,
    /// Camera prims.
    pub cameras: bool,
    /// The render settings.
    pub settings: bool,
}

impl Changed {
    /// Whether anything changed.
    pub fn any(&self) -> bool {
        self.geometry || self.materials || self.lights || self.fog || self.cameras || self.settings
    }
}

/// The mesh changes that move geometry.
const GEOMETRY: Dirty = Dirty::POINTS
    .union(Dirty::TOPOLOGY)
    .union(Dirty::NORMALS)
    .union(Dirty::MATERIAL_SLOTS)
    .union(Dirty::ATTRIBUTES)
    .union(Dirty::TRANSFORM)
    .union(Dirty::VISIBILITY);

/// A light prim, decoded.
#[derive(Clone, Debug, PartialEq)]
enum Emitter {
    /// A standard light, and its gobo when it has one on.
    Standard(standard::Light, Option<Gobo>),
    /// A rim light.
    Rim(Rim),
}

impl Emitter {
    /// `prim` as a light Pixel renders (none when it is of another type).
    fn decode(prim: &Declared) -> Option<Emitter> {
        standard::Light::decode(prim)
            .map(|light| Emitter::Standard(light, Gobo::decode(prim)))
            .or_else(|| Rim::decode(prim).map(Emitter::Rim))
    }
}

/// The prims Pixel Render reads, by id.
#[derive(Clone, Debug, Default)]
pub struct Store {
    meshes: BTreeMap<PrimId, RenderMesh>,
    instancers: BTreeMap<PrimId, Instancer>,
    lights: BTreeMap<PrimId, Emitter>,
    fogs: BTreeMap<PrimId, Fog>,
    cameras: BTreeMap<PrimId, Camera>,
    materials: BTreeMap<PrimId, PixelMaterial>,
    settings: RenderSettings,
}

impl Store {
    /// An empty scene.
    pub fn new() -> Store {
        Store::default()
    }

    /// Apply `update`; says what it changed.
    pub fn apply(&mut self, update: SceneUpdate) -> Changed {
        let mut changed = Changed::default();
        for id in update.removed {
            self.remove(id, &mut changed);
        }
        for (id, prim) in update.added {
            self.remove(id, &mut changed);
            self.insert(id, prim, Dirty::ALL, &mut changed);
        }
        for change in update.changed {
            self.insert(change.id, change.prim, change.dirty, &mut changed);
        }
        if let Some(settings) = update.settings {
            changed.settings |= settings != self.settings;
            self.settings = settings;
        }
        changed
    }

    fn remove(&mut self, id: PrimId, changed: &mut Changed) {
        changed.geometry |= self.meshes.remove(&id).is_some();
        changed.geometry |= self.instancers.remove(&id).is_some();
        changed.lights |= self.lights.remove(&id).is_some();
        changed.fog |= self.fogs.remove(&id).is_some();
        changed.cameras |= self.cameras.remove(&id).is_some();
        changed.materials |= self.materials.remove(&id).is_some();
    }

    fn insert(&mut self, id: PrimId, prim: Prim, dirty: Dirty, changed: &mut Changed) {
        match prim {
            Prim::Mesh(mesh) => {
                changed.geometry |= dirty.intersects(GEOMETRY);
                self.meshes.insert(id, mesh);
            }
            Prim::Instancer(instancer) => {
                changed.geometry = true;
                self.instancers.insert(id, instancer);
            }
            Prim::Declared(prim) => {
                if let Some(fog) = Fog::decode(&prim) {
                    changed.fog = true;
                    self.fogs.insert(id, fog);
                } else if let Some(light) = Emitter::decode(&prim) {
                    changed.lights = true;
                    self.lights.insert(id, light);
                }
                // Hosts send no types Pixel does not declare.
            }
            Prim::Camera(camera) => {
                changed.cameras = true;
                self.cameras.insert(id, camera);
            }
            Prim::Material(material) => {
                changed.materials = true;
                let now = crate::material::read(&material.network)
                    .unwrap_or_else(|| PixelMaterial::Surface(PixelSurface::default()));
                // Another color attribute retints the meshes that name it;
                // turning to or from a volume moves their triangles.
                let before = self.materials.get(&id);
                let reshaped = match (before, &now) {
                    (None, PixelMaterial::Surface(s)) => !s.attribute.is_empty(),
                    (Some(PixelMaterial::Surface(a)), PixelMaterial::Surface(b)) => {
                        a.attribute != b.attribute
                    }
                    (Some(PixelMaterial::Volume(_)), PixelMaterial::Volume(_)) => false,
                    _ => true,
                };
                if reshaped {
                    changed.geometry |= self.meshes.values().any(|m| m.materials.contains(&id));
                }
                self.materials.insert(id, now);
            }
            // Pixel Surfaces read no images.
            Prim::Texture(_) => {}
        }
    }

    /// The render settings last sent.
    pub fn settings(&self) -> &RenderSettings {
        &self.settings
    }

    /// The scene's fog: the first fog prim's, its values in range.
    pub fn fog(&self) -> Option<Fog> {
        let f = self.fogs.values().next()?;
        let at_least_0 = |v: f32| if v.is_finite() { v.max(0.0) } else { 0.0 };
        Some(Fog {
            density: at_least_0(f.density),
            color: f.color.map(at_least_0),
            start: at_least_0(f.start),
            height: if f.height.is_finite() { f.height } else { 0.0 },
            falloff: at_least_0(f.falloff),
            inscatter: f.inscatter,
            anisotropy: if f.anisotropy.is_finite() {
                f.anisotropy.clamp(-0.9, 0.9)
            } else {
                0.0
            },
        })
    }

    /// Camera prim `id`.
    pub fn camera(&self, id: PrimId) -> Option<Camera> {
        self.cameras.get(&id).cloned()
    }

    /// The surface materials, by their prim's slot (slots no surface holds
    /// take the default surface).
    pub fn materials(&self) -> Vec<PixelSurface> {
        let len = self.materials.keys().map(|id| id.slot as usize + 1).max();
        let mut out = vec![PixelSurface::default(); len.unwrap_or(0)];
        for (id, material) in &self.materials {
            if let PixelMaterial::Surface(surface) = material {
                out[id.slot as usize] = surface.clone();
            }
        }
        out
    }

    /// The material in slot `slot`, if one is sent.
    fn material(&self, slot: u32) -> Option<&PixelMaterial> {
        self.materials
            .iter()
            .find(|(id, _)| id.slot == slot)
            .map(|(_, m)| m)
    }

    /// The meshes: each mesh's surface triangles by material, then each
    /// instancer's copies of its prototype.
    pub fn meshes(&self) -> Vec<Mesh> {
        self.pieces()
            .into_iter()
            .filter(|m| !matches!(self.material(m.material), Some(PixelMaterial::Volume(_))))
            .collect()
    }

    /// The volumes: the triangles naming a Pixel Volume, by material, as
    /// [`Self::meshes`] orders them.
    pub fn volumes(&self) -> Vec<Volume> {
        self.pieces()
            .into_iter()
            .filter_map(|mesh| match self.material(mesh.material) {
                Some(PixelMaterial::Volume(medium)) => Some(Volume {
                    medium: medium.clone(),
                    mesh,
                }),
                _ => None,
            })
            .collect()
    }

    /// Every mesh's and instance's triangles, one piece a material.
    fn pieces(&self) -> Vec<Mesh> {
        let attribute = |slot: u32| -> &str {
            match self.material(slot) {
                Some(PixelMaterial::Surface(s)) => &s.attribute,
                _ => "",
            }
        };
        let mut out = Vec::new();
        for mesh in self.meshes.values() {
            split(mesh, mesh.transform, &attribute, &mut out);
        }
        for instancer in self.instancers.values().filter(|i| i.visible) {
            let Some(prototype) = self.meshes.get(&instancer.prototype) else {
                continue;
            };
            for rows in instancer.transforms.iter() {
                split(prototype, placement(rows), &attribute, &mut out);
            }
        }
        out
    }

    /// The lights but domes and rims, the dome all dome lights add up to
    /// and the rims all rim lights add up to.
    pub fn lights(&self) -> Lighting {
        let mut lighting = Lighting::default();
        for emitter in self.lights.values() {
            let (light, gobo) = match emitter {
                Emitter::Standard(light, gobo) => (light, *gobo),
                Emitter::Rim(rim) => {
                    let color = emission_color(&rim.emission, rim.color, rim.color_temperature);
                    let rims = &mut lighting.rims;
                    add(
                        match rim.side {
                            RimSide::Lit => &mut rims.lit,
                            RimSide::All => &mut rims.all,
                        },
                        color,
                        rim.intensity * rim.exposure.exp2(),
                    );
                    continue;
                }
            };
            let color = emission_color(&light.emission, light.color, light.color_temperature);
            let k = light.intensity * light.exposure.exp2();
            let shape = match light.shape {
                Shape::Rect { width, height } => LightShape::Rect { width, height },
                Shape::Disk { radius } => LightShape::Disk { radius },
                Shape::Distant { angle } => LightShape::Distant { angle },
                Shape::Spot {
                    radius,
                    cone_angle,
                    cone_softness,
                } => LightShape::Spot {
                    radius,
                    cone_angle,
                    cone_softness,
                },
                Shape::Dome { ground, .. } => {
                    let dome = lighting.dome.get_or_insert_with(Dome::default);
                    add(&mut dome.above, color, k);
                    add(&mut dome.below, ground.unwrap_or(color), k);
                    dome.visible |= light.visible;
                    continue;
                }
            };
            lighting.lights.push(Light {
                shape,
                color,
                intensity: light.intensity,
                exposure: light.exposure,
                normalize: light.normalize,
                transform: light.transform,
                gobo,
            });
        }
        lighting
    }
}

/// Add `c` × `k` to `sum`.
fn add(sum: &mut [f32; 3], c: [f32; 3], k: f32) {
    for (s, e) in sum.iter_mut().zip(c) {
        *s += e * k;
    }
}

/// `mesh` at `transform` as Pixel meshes, one a material its triangles
/// name (one mesh sharing its arrays when they name one), each tinted by
/// the attribute its material reads (`attribute` of a material's slot).
fn split<'a>(
    mesh: &RenderMesh,
    transform: Matrix,
    attribute: &dyn Fn(u32) -> &'a str,
    out: &mut Vec<Mesh>,
) {
    let material = |t: usize| -> u32 {
        mesh.material_slots
            .get(t)
            .and_then(|&s| mesh.materials.get(usize::from(s)))
            .map_or(u32::MAX, |id| id.slot)
    };
    let tints = |material: u32, triangles: &mut dyn Iterator<Item = usize>| {
        crate::tint::corners(mesh, attribute(material), triangles)
    };
    let of = |material, triangles, normals, tints| Mesh {
        points: mesh.points.clone(),
        triangles,
        normals,
        tints,
        transform,
        material,
        id: mesh.id,
        visible: mesh.visible,
    };
    let count = mesh.triangles.len();
    let first = material(0);
    if (1..count).all(|t| material(t) == first) {
        let tinted = tints(first, &mut (0..count));
        out.push(of(
            first,
            mesh.triangles.clone(),
            mesh.normals.clone(),
            tinted,
        ));
        return;
    }
    let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for t in 0..count {
        groups.entry(material(t)).or_default().push(t);
    }
    let normals = mesh.normals.as_ref().filter(|n| n.len() == count * 3);
    for (m, triangles) in groups {
        let corners = normals.map(|n| -> Arc<[[f32; 3]]> {
            triangles
                .iter()
                .flat_map(|&t| [n[t * 3], n[t * 3 + 1], n[t * 3 + 2]])
                .collect()
        });
        let tinted = tints(m, &mut triangles.iter().copied());
        let triangles: Arc<[[u32; 3]]> = triangles.iter().map(|&t| mesh.triangles[t]).collect();
        out.push(of(m, triangles, corners, tinted));
    }
}

/// An instancer's placement (its matrix's first three rows, row-major) as
/// a column-major matrix.
fn placement(rows: &[[f32; 4]; 3]) -> Matrix {
    let mut m = [[0.0; 4]; 4];
    for (r, row) in rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            m[c][r] = v;
        }
    }
    m[3][3] = 1.0;
    m
}

/// The color an emitter of `emission` shows at intensity 1, linear ACEScg:
/// its `color`, or its blackbody's (at `kelvin`) or illuminant's at
/// luminance 1 (an illuminant no spectrum has: its color).
fn emission_color(emission: &Emission, color: [f32; 3], kelvin: f32) -> [f32; 3] {
    let rgb = match emission {
        Emission::Color => return color,
        Emission::Temperature => {
            let kelvin = f64::from(kelvin);
            if kelvin.is_nan() || kelvin <= 0.0 {
                return [0.0; 3];
            }
            crate::spectrum::blackbody(kelvin)
        }
        Emission::Illuminant(name) => {
            let Some(rgb) = crate::spectrum::illuminant(name) else {
                return color;
            };
            rgb
        }
    };
    rgb.map(|v| v as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::PixelVolume;
    use kinogaki_render_api::{Change, IDENTITY, Material, MaterialNetwork};

    fn id(slot: u32) -> PrimId {
        PrimId {
            slot,
            generation: 0,
        }
    }

    fn mesh(slots: Vec<u16>, materials: Vec<PrimId>) -> Prim {
        let triangles = slots.len();
        Prim::Mesh(RenderMesh {
            motion: kinogaki_render_api::Motion::none(),
            points: Arc::from(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            triangles: Arc::from(vec![[0u32, 1, 2]; triangles]),
            normals: None,
            uvs: None,
            material_slots: Arc::from(slots),
            materials: Arc::from(materials),
            attributes: Arc::from([]),
            transform: IDENTITY,
            visible: true,
            id: 7,
            path: Arc::from("/mesh"),
        })
    }

    fn material(color: [f32; 3]) -> Prim {
        Prim::Material(Material {
            network: Arc::new(crate::material::network(color, "Cd")),
            path: Arc::from("/m"),
        })
    }

    #[test]
    fn updates_say_what_they_change() {
        let mut store = Store::new();
        let mut first = SceneUpdate::new();
        first.added.push((id(0), material([1.0, 0.0, 0.0])));
        first.added.push((id(1), mesh(vec![0, 0], vec![id(0)])));
        let changed = store.apply(first);
        assert!(changed.geometry && changed.materials && !changed.lights);
        assert_eq!(store.meshes().len(), 1);
        assert_eq!(store.meshes()[0].material, 0);
        assert_eq!(store.materials()[0].color, [1.0, 0.0, 0.0]);

        // A material edit and a camera move leave the geometry.
        let mut edit = SceneUpdate::new();
        edit.changed
            .push(Change::new(id(0), Dirty::PARAMS, material([0.0, 1.0, 0.0])));
        edit.added.push((id(2), Prim::Camera(Camera::default())));
        let changed = store.apply(edit);
        assert!(!changed.geometry && changed.materials && changed.cameras);
        assert_eq!(store.materials()[0].color, [0.0, 1.0, 0.0]);

        // A transform does.
        let mut moved = SceneUpdate::new();
        moved.changed.push(Change::new(
            id(1),
            Dirty::TRANSFORM,
            mesh(vec![0, 0], vec![id(0)]),
        ));
        assert!(store.apply(moved).geometry);
        // So does an attribute change: the tints are the vertices'.
        let mut attrs = SceneUpdate::new();
        attrs.changed.push(Change::new(
            id(1),
            Dirty::ATTRIBUTES,
            mesh(vec![0, 0], vec![id(0)]),
        ));
        assert!(store.apply(attrs).geometry);
        // A material naming another attribute retints its meshes; a new
        // color alone does not.
        let mut renamed = SceneUpdate::new();
        let tone = Prim::Material(Material {
            network: Arc::new(crate::material::network([0.0, 1.0, 0.0], "tone")),
            path: Arc::from("/m"),
        });
        renamed
            .changed
            .push(Change::new(id(0), Dirty::PARAMS, tone));
        assert!(store.apply(renamed).geometry);
        let mut recolored = SceneUpdate::new();
        let red = Prim::Material(Material {
            network: Arc::new(crate::material::network([1.0, 0.0, 0.0], "tone")),
            path: Arc::from("/m"),
        });
        recolored
            .changed
            .push(Change::new(id(0), Dirty::PARAMS, red));
        let changed = store.apply(recolored);
        assert!(changed.materials && !changed.geometry);
    }

    #[test]
    fn meshes_carry_the_attribute_their_material_reads() {
        let mut store = Store::new();
        let mut update = SceneUpdate::new();
        update.added.push((id(0), material([1.0; 3])));
        update.added.push((
            id(2),
            Prim::Material(Material {
                network: Arc::new(crate::material::network([1.0; 3], "")),
                path: Arc::from("/plain"),
            }),
        ));
        let Prim::Mesh(mut cd) = mesh(vec![0, 1], vec![id(0), id(2)]) else {
            unreachable!()
        };
        cd.attributes = Arc::from(vec![kinogaki_render_api::MeshAttribute {
            name: Arc::from("Cd"),
            domain: kinogaki_render_api::AttributeDomain::Triangle,
            components: 3,
            color: true,
            integer: false,
            values: Arc::from(vec![0.5, 0.25, 0.0, 0.0, 0.0, 1.0]),
        }]);
        update.added.push((id(1), Prim::Mesh(cd)));
        store.apply(update);
        let meshes = store.meshes();
        // Triangle 0 reads Cd; triangle 1's material reads nothing.
        assert_eq!(meshes[0].material, 0);
        assert_eq!(meshes[0].tints.as_deref(), Some(&[[0.5, 0.25, 0.0]; 3][..]));
        assert_eq!(meshes[1].material, 2);
        assert_eq!(meshes[1].tints, None);
    }

    #[test]
    fn meshes_split_by_material_and_unnamed_ones_take_the_default() {
        let mut store = Store::new();
        let mut update = SceneUpdate::new();
        update.added.push((id(3), material([0.5; 3])));
        update
            .added
            .push((id(1), mesh(vec![0, 1, 0, 9], vec![id(3), id(5)])));
        store.apply(update);
        let meshes = store.meshes();
        let by: Vec<(u32, usize)> = meshes
            .iter()
            .map(|m| (m.material, m.triangles.len()))
            .collect();
        // Slot 1 names material 5 (not sent); slot 9 names none.
        assert_eq!(by, vec![(3, 2), (5, 1), (u32::MAX, 1)]);
        let materials = store.materials();
        assert_eq!(materials.len(), 4);
        assert_eq!(materials[3].color, [0.5; 3]);
        assert_eq!(materials[0], PixelSurface::default());
    }

    #[test]
    fn domes_add_up_and_emitters_show_their_emission() {
        let mut store = Store::new();
        let mut update = SceneUpdate::new();
        let light = |light: standard::Light| Prim::Declared(light.declared());
        let dome = |intensity, visible| {
            light(standard::Light {
                shape: Shape::Dome {
                    image: None,
                    ground: None,
                },
                color: [1.0, 0.5, 0.25],
                intensity,
                visible,
                ..standard::Light::default()
            })
        };
        update.added.push((id(0), dome(1.0, false)));
        update.added.push((id(1), dome(2.0, true)));
        update.added.push((
            id(2),
            light(standard::Light {
                emission: Emission::Temperature,
                color_temperature: 2700.0,
                ..standard::Light::default()
            }),
        ));
        // A gradient dome: its ground color below the horizon, times its
        // intensity and exposure.
        update.added.push((
            id(3),
            light(standard::Light {
                shape: Shape::Dome {
                    image: None,
                    ground: Some([0.0, 1.0, 0.0]),
                },
                color: [0.5; 3],
                intensity: 2.0,
                exposure: -1.0,
                visible: false,
                ..standard::Light::default()
            }),
        ));
        // Rims add up by side, in their emission's color.
        let rim = |side, intensity| {
            let rim = Rim {
                side,
                emission: Emission::Color,
                color: [1.0, 0.5, 0.0],
                intensity,
                ..Rim::default()
            };
            Prim::Declared(rim.declared())
        };
        update.added.push((id(4), rim(RimSide::Lit, 0.5)));
        update.added.push((id(5), rim(RimSide::Lit, 0.25)));
        update.added.push((id(6), rim(RimSide::All, 2.0)));
        store.apply(update);
        let lighting = store.lights();
        let dome = lighting.dome.expect("a dome");
        assert_eq!(dome.above, [3.5, 2.0, 1.25]);
        assert_eq!(dome.below, [3.0, 2.5, 0.75]);
        assert!(dome.visible);
        assert_eq!(lighting.rims.lit, [0.75, 0.375, 0.0]);
        assert_eq!(lighting.rims.all, [2.0, 1.0, 0.0]);
        // A warm blackbody: red over blue, at luminance 1.
        let lights = lighting.lights;
        assert_eq!(lights.len(), 1, "domes and rims are not direct lights");
        let c = lights[0].color;
        assert!(c[0] > c[2], "{c:?}");
        let y = crate::dither::luminance(c);
        assert!((y - 1.0).abs() < 0.02, "luminance {y}");
    }

    #[test]
    fn distant_lights_and_spots_carry_their_gobo() {
        let dapple = Gobo {
            scale: 2.0,
            seed: 3,
            ..Gobo::default()
        };
        let sun = standard::Light {
            shape: Shape::Distant { angle: 0.0 },
            ..standard::Light::default()
        };
        let mut store = Store::new();
        let mut update = SceneUpdate::new();
        let on = Gobo::apply(Some(dapple), sun.declared());
        update.added.push((id(0), Prim::Declared(on)));
        let off = Gobo::apply(None, sun.declared());
        update.added.push((id(1), Prim::Declared(off)));
        let changed = store.apply(update);
        assert!(changed.lights && !changed.fog);
        let gobos: Vec<_> = store.lights().lights.iter().map(|l| l.gobo).collect();
        assert_eq!(gobos, [Some(dapple), None]);
    }

    #[test]
    fn the_first_fog_prim_is_the_scenes() {
        let mut store = Store::new();
        let mut update = SceneUpdate::new();
        let thick = Fog {
            density: 2.0,
            ..Fog::default()
        };
        update.added.push((id(3), Prim::Declared(thick.declared())));
        let changed = store.apply(update);
        assert!(changed.fog && !changed.lights && !changed.geometry);
        assert_eq!(store.fog(), Some(thick));
        let mut gone = SceneUpdate::new();
        gone.removed.push(id(3));
        assert!(store.apply(gone).fog);
        assert_eq!(store.fog(), None);
    }

    #[test]
    fn instancers_copy_their_prototype() {
        let mut store = Store::new();
        let mut update = SceneUpdate::new();
        update.added.push((id(0), mesh(vec![0], Vec::new())));
        let row = |x| {
            [
                [1.0, 0.0, 0.0, x],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        };
        update.added.push((
            id(1),
            Prim::Instancer(Instancer {
                motion: kinogaki_render_api::Motion::none(),
                prototype: id(0),
                transforms: Arc::from(vec![row(2.0), row(4.0)]),
                visible: true,
            }),
        ));
        store.apply(update);
        let meshes = store.meshes();
        assert_eq!(meshes.len(), 3);
        assert_eq!(meshes[2].transform[3], [4.0, 0.0, 0.0, 1.0]);
        // A foreign material renders as the default surface.
        let mut foreign = SceneUpdate::new();
        foreign.added.push((
            id(4),
            Prim::Material(Material {
                network: Arc::new(MaterialNetwork {
                    engine: Arc::from("spectre"),
                    ..MaterialNetwork::default()
                }),
                path: Arc::from("/s"),
            }),
        ));
        store.apply(foreign);
        assert_eq!(store.materials()[4], PixelSurface::default());
    }

    #[test]
    fn volume_triangles_leave_the_meshes() {
        let mut store = Store::new();
        let mut update = SceneUpdate::new();
        update.added.push((id(0), material([1.0; 3])));
        update
            .added
            .push((id(1), mesh(vec![0, 1, 1], vec![id(0), id(2)])));
        store.apply(update);
        assert_eq!(store.meshes().len(), 2);
        assert!(store.volumes().is_empty());
        // Slot 2 becomes a volume: its triangles leave the meshes.
        let fog = || {
            Prim::Material(Material {
                network: Arc::new(crate::material::volume::network(&PixelVolume {
                    density: 2.0,
                    ..PixelVolume::default()
                })),
                path: Arc::from("/fog"),
            })
        };
        let mut add = SceneUpdate::new();
        add.added.push((id(2), fog()));
        assert!(store.apply(add).geometry);
        let meshes = store.meshes();
        assert_eq!((meshes.len(), meshes[0].material), (1, 0));
        let volumes = store.volumes();
        assert_eq!(volumes.len(), 1);
        assert_eq!(volumes[0].mesh.triangles.len(), 2);
        assert_eq!(volumes[0].medium.density, 2.0);
        assert_eq!(store.materials()[2], PixelSurface::default());
        // A volume's inputs change its medium, not the geometry.
        let mut thicker = SceneUpdate::new();
        thicker
            .changed
            .push(Change::new(id(2), Dirty::PARAMS, fog()));
        let changed = store.apply(thicker);
        assert!(changed.materials && !changed.geometry);
        // Back to a surface: the triangles return.
        let mut back = SceneUpdate::new();
        back.changed
            .push(Change::new(id(2), Dirty::PARAMS, material([0.5; 3])));
        assert!(store.apply(back).geometry);
        assert!(store.volumes().is_empty());
        assert_eq!(store.meshes().len(), 2);
    }
}
