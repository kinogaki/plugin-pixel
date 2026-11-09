//! [`Sync`]'s declared prims as Pixel's lights: the domes summed into the
//! ambient light (those a portal names apart, lit through their portals),
//! the portals, the rims by their side, the first fog, and a geometry
//! light as a sphere of its meshes' area at their centre, with its slot
//! ([`crate::shadowless`]).
use super::Sync;
use crate::lights::{self, RIM_LIGHT};
use crate::render::{Ambient, Fog, Portal, Rims};
use crate::shadowless::NO_SLOT;
use kinogaki_render_api::geometry::{Diagnostic, Vec3, Xform};
use kinogaki_render_api::standard::{self, LightShape as LightKind};
use kinogaki_render_api::{Prim, PrimId, Topology};
use std::collections::BTreeSet;

impl Sync {
    /// The lights, domes, portals, rims and fog again.
    pub(super) fn read_lights(&mut self) {
        let (mut lights, mut ambient, mut rims, mut fog) =
            (Vec::new(), Ambient::default(), Rims::default(), None);
        let (mut glows, mut portals, mut said) = (Vec::new(), Vec::new(), Vec::new());
        let declared: Vec<(PrimId, &kinogaki_render_api::Declared)> = (self.prims.iter())
            .filter_map(|(id, prim)| match prim {
                Prim::Declared(d) => Some((*id, d)),
                _ => None,
            })
            .collect();
        // The domes some portal names, and the portals.
        let mut served = BTreeSet::new();
        for (_, prim) in &declared {
            let Some(light) = standard::Light::decode(prim) else {
                continue;
            };
            if let LightKind::Portal { width, height, .. } = light.shape
                && let Some(&dome) = prim.members("dome").prims().first()
            {
                served.insert(dome);
                portals.push(portal_of(&light.xform.at(0.0), width, height));
            }
        }
        for &(id, prim) in &declared {
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
            match light.shape {
                LightKind::Dome { ground, .. } => {
                    let radiance = light.radiance().map(|c| c * light.diffuse.max(0.0));
                    let scale = light.intensity * light.exposure.exp2() * light.diffuse.max(0.0);
                    let below = ground.map_or(radiance, |g| g.map(|c| c * scale));
                    if served.contains(&id) {
                        add(&mut ambient.through_sky, radiance);
                        add(&mut ambient.through_ground, below);
                    } else {
                        add(&mut ambient.sky, radiance);
                        add(&mut ambient.ground, below);
                        ambient.seen |= light.camera;
                    }
                    continue;
                }
                LightKind::Portal { .. } => continue,
                LightKind::Geometry { .. } => {
                    let named = prim.members("geometry");
                    let (area, centre, copies) = self.area_of(named.prims());
                    if copies {
                        said.push(Diagnostic::warning(
                            "a geometry light names copies: Pixel lights by their meshes' own shapes alone",
                        ));
                    }
                    let slot = self.shadowless.slot(id);
                    lights.extend(lights::mesh_light(&light, (area, centre), slot));
                    continue;
                }
                _ => {}
            }
            let texture = |name: &str| match self.prims.get(&prim.textures.get(name)?.prim()) {
                Some(Prim::Texture(read)) => Some(read.clone()),
                _ => None,
            };
            if let Some((light, glow)) = lights::light_of(prim, &texture, NO_SLOT) {
                lights.push(light);
                glows.extend(glow);
            }
        }
        if self.shadowless.dropped > 0 {
            said.push(Diagnostic::warning(format!(
                "{} geometry lights past Pixel's {}: their meshes shadow them",
                self.shadowless.dropped,
                crate::shadowless::MOST
            )));
        }
        self.problems.lights = said;
        (
            self.scene.lights,
            self.scene.glows,
            self.scene.ambient,
            self.scene.portals,
            self.scene.rims,
            self.scene.fog,
        ) = (lights, glows, ambient, portals, rims, fog);
    }

    /// The world area and area-weighted centre of the visible Shape prims
    /// among `prims` (as each stands at the shutter's opening), and whether
    /// `prims` names an Instancer, whose copies are left out.
    fn area_of(&self, prims: &[PrimId]) -> (f32, [f32; 3], bool) {
        let (mut area, mut moment, mut copies) = (0.0f64, Vec3::ZERO, false);
        for prim in prims {
            let shape = match self.prims.get(prim) {
                Some(Prim::Shape(shape)) if shape.visibility().visible => shape,
                Some(Prim::Instancer(_)) => {
                    copies = true;
                    continue;
                }
                _ => continue,
            };
            let Topology::Triangles(triangles) = shape.topology() else {
                continue;
            };
            let at = shape.xform().at(0.0);
            let points = shape.points();
            let world = |i: u32| at.apply_point(Vec3::from(points[i as usize].map(f64::from)));
            for [a, b, c] in triangles.as_slice().iter().map(|t| t.map(world)) {
                let piece = 0.5 * (b - a).cross(c - a).length();
                area += piece;
                moment += (a + b + c) * (piece / 3.0);
            }
        }
        let centre = if area > 0.0 {
            moment * (1.0 / area)
        } else {
            Vec3::ZERO
        };
        (area as f32, centre.to_f32(), copies)
    }
}

/// A `width` × `height` portal placed by `at`, in the world.
fn portal_of(at: &Xform, width: f32, height: f32) -> Portal {
    let axis = |v: Vec3| at.apply_vector(v).to_f32();
    let passes = at.apply_vector(Vec3::new(0.0, 0.0, -1.0));
    Portal {
        centre: at.apply_point(Vec3::ZERO).to_f32(),
        u: axis(Vec3::new(0.5 * f64::from(width), 0.0, 0.0)),
        v: axis(Vec3::new(0.0, 0.5 * f64::from(height), 0.0)),
        passes: passes
            .normalized()
            .unwrap_or(Vec3::new(0.0, 0.0, -1.0))
            .to_f32(),
    }
}

/// Adds `b` into `a`.
fn add(a: &mut [f32; 3], b: [f32; 3]) {
    *a = std::array::from_fn(|k| a[k] + b[k]);
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
