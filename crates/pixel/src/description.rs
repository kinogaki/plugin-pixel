//! Pixel Render as hosts see it before making one: [`description`] gives
//! its settings, the prim types it declares, its material vocabulary,
//! presets, outputs and preview setup, and how its materials look
//! elsewhere ([`tint`], [`PixelMaterials`]), as data a host builds its UI
//! and files from.

use std::sync::Arc;

use kinogaki_render_api::{
    AovSpec, EngineDescription, MaterialNetwork, MaterialPreset, MaterialRules, PreviewSetup,
    PreviewSurface, PrimKind, Tint, standard,
};

use crate::aov::Aov;
use crate::material::volume::input as volume_input;
use crate::material::{self, COLOR, ID, SURFACE, VOLUME};
use crate::scene::{PixelMaterial, PixelSurface};

/// Pixel Render, described.
pub fn description() -> EngineDescription {
    EngineDescription {
        id: Arc::from(ID),
        label: Arc::from("Pixel"),
        prims: vec![
            PrimKind::Mesh,
            PrimKind::Instancer,
            PrimKind::Camera,
            PrimKind::Material,
        ],
        // UsdLux's lights, the fog and the rim light; a gobo on distant
        // lights and spots.
        types: [
            standard::lights(),
            vec![standard::fog(), crate::rim::prim_type()],
        ]
        .concat(),
        applied: vec![crate::gobo::schema()],
        setting_groups: crate::settings::groups(),
        settings: crate::settings::descriptors(),
        material_nodes: material::node_types(),
        material_presets: presets(),
        // Every output, the beauty first; all asked for by default (each
        // costs a film-size plane).
        aovs: Aov::ALL
            .into_iter()
            .map(|a| AovSpec::new(a.output(), true))
            .collect(),
        realtime: true,
        preview: PreviewSetup {
            // The defaults: swatches are pixel art too.
            settings: Vec::new(),
            floor: material::network([0.18; 3], ""),
        },
        materials: Arc::new(PixelMaterials),
    }
}

/// What Pixel materials are to others: [`tint`], [`material::is_volume`],
/// [`flat`], [`default_material`], [`preview_surface`] and
/// [`from_preview_surface`].
pub struct PixelMaterials;

impl MaterialRules for PixelMaterials {
    fn tint(&self, network: &MaterialNetwork) -> Option<Tint> {
        tint(network)
    }

    fn volume(&self, network: &MaterialNetwork) -> bool {
        material::is_volume(network)
    }

    fn flat(&self, tint: &Tint) -> MaterialNetwork {
        flat(tint)
    }

    fn default_material(&self) -> MaterialNetwork {
        default_material()
    }

    fn preview_surface(&self, network: &MaterialNetwork) -> PreviewSurface {
        preview_surface(network)
    }

    fn import_preview_surface(&self, surface: &PreviewSurface) -> MaterialNetwork {
        from_preview_surface(surface)
    }
}

/// How a Pixel material shows elsewhere: its surface's color, times the
/// attribute it reads when it names one (translucency is Pixel's own and
/// shows nowhere else); a volume's scatter color.
pub fn tint(network: &MaterialNetwork) -> Option<Tint> {
    let s = match material::read(network)? {
        PixelMaterial::Surface(s) => s,
        PixelMaterial::Volume(v) => return Some(Tint::Color(v.scatter_color)),
    };
    Some(if s.attribute.is_empty() {
        Tint::Color(s.color)
    } else {
        Tint::Modulated {
            color: s.color,
            attribute: s.attribute,
        }
    })
}

/// An opaque Pixel Surface showing `tint`: what Pixel Render renders
/// another engine's material as. A bare attribute's tint is white times it
/// (white, not black, where a mesh lacks it).
pub fn flat(tint: &Tint) -> MaterialNetwork {
    match tint {
        Tint::Color(c) => material::network(*c, ""),
        Tint::Attribute(name) => material::network([1.0; 3], name),
        Tint::Modulated { color, attribute } => material::network(*color, attribute),
    }
}

/// The default Pixel Surface: what renders where no material is named
/// (it reads no attribute: a host sends such triangles none).
pub fn default_material() -> MaterialNetwork {
    let s = PixelSurface::default();
    material::network(s.color, &s.attribute)
}

/// A Pixel material as a preview surface: its color, diffuse.
pub fn preview_surface(network: &MaterialNetwork) -> PreviewSurface {
    PreviewSurface {
        diffuse: material::surface(network).map(|s| s.color.map(f64::from)),
        ..PreviewSurface::default()
    }
}

/// A Pixel Surface of a preview surface's diffuse color.
pub fn from_preview_surface(surface: &PreviewSurface) -> MaterialNetwork {
    surface.diffuse.map_or_else(default_material, |c| {
        material::network(c.map(|v| v as f32), material::DEFAULT_ATTRIBUTE)
    })
}

/// The material presets, in the order hosts list them.
pub fn presets() -> Vec<MaterialPreset> {
    let words = |aliases: &[&str]| aliases.iter().map(|&a| Arc::from(a)).collect();
    let preset = |name, aliases: &[&str], color: [f64; 3]| MaterialPreset {
        name: Arc::from(name),
        aliases: words(aliases),
        surface: Arc::from(SURFACE),
        params: Vec::new(),
        values: vec![(Arc::from(COLOR), color)],
    };
    let volume = |name, aliases: &[&str], values: &[(&str, [f64; 3])]| MaterialPreset {
        name: Arc::from(name),
        aliases: words(aliases),
        surface: Arc::from(VOLUME),
        params: Vec::new(),
        values: values.iter().map(|&(n, v)| (Arc::from(n), v)).collect(),
    };
    vec![
        preset("Grass", &["green", "leaf", "foliage"], [0.12, 0.32, 0.06]),
        preset("Stone", &["rock", "grey", "gray"], [0.32, 0.31, 0.3]),
        preset(
            "Clay",
            &["terracotta", "brick", "orange"],
            [0.55, 0.22, 0.1],
        ),
        // The fog material: even fog filling a box or any closed shape,
        // its edges sharp (the scene's Fog object is the global kind).
        volume(
            "Fog",
            &["fog material", "bounded fog", "haze", "volume", "smoke"],
            &[
                (volume_input::DENSITY, [0.3; 3]),
                (volume_input::SCATTER_COLOR, [0.8, 0.82, 0.88]),
                (volume_input::ANISOTROPY, [0.3; 3]),
                (volume_input::EDGE_FALLOFF, [0.0; 3]),
                (volume_input::NOISE_AMOUNT, [0.0; 3]),
            ],
        ),
        // Low mist for a box over the ground: forward scattering for
        // light shafts, faded in from its faces.
        volume(
            "Mist",
            &["haze", "god rays", "volume"],
            &[
                (volume_input::DENSITY, [0.15; 3]),
                (volume_input::SCATTER_COLOR, [0.85, 0.88, 0.95]),
                (volume_input::ANISOTROPY, [0.5; 3]),
                (volume_input::EDGE_FALLOFF, [0.5; 3]),
            ],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinogaki_render_api::aov::standard as aov;

    #[test]
    fn the_description_lists_pixel_surface_and_every_output() {
        let d = description();
        assert_eq!((&*d.id, &*d.label), ("pixel", "Pixel"));
        assert_eq!(d.surfaces().count(), 2);
        assert_eq!(&*d.node_type(VOLUME).unwrap().label, "Pixel Volume");
        assert_eq!(&*d.node_type(SURFACE).unwrap().label, "Pixel Surface");
        assert_eq!(d.aovs[0].aov, aov::beauty());
        assert_eq!(d.aov("occlusion"), aov::get(aov::OCCLUSION).as_ref());
        assert_eq!(d.aov("volume"), aov::get(aov::VOLUME).as_ref());
        assert!(d.aov("CryptoObject").is_none());
        assert!(d.realtime);
        for p in &d.material_presets {
            assert!(d.node_type(&p.surface).is_some(), "{}", p.name);
        }
        // Fog and Mist are the fog materials: Pixel Volumes.
        for name in ["Fog", "Mist"] {
            let p = d.material_presets.iter().find(|p| &*p.name == name);
            assert_eq!(p.map(|p| &*p.surface), Some(VOLUME));
        }
    }

    #[test]
    fn the_description_declares_the_standard_types_the_rim_and_the_gobo() {
        let d = description();
        let ids: Vec<&str> = d.types.iter().map(|t| &*t.id).collect();
        assert_eq!(
            ids,
            [
                standard::AREA_LIGHT,
                standard::DOME_LIGHT,
                standard::DISTANT_LIGHT,
                standard::SPOT_LIGHT,
                standard::FOG,
                crate::rim::ID,
            ]
        );
        assert!(d.prims.iter().all(|k| PrimKind::STRUCTURAL.contains(k)));
        let (groups, params) = d.schema(standard::SPOT_LIGHT).unwrap();
        assert_eq!(&*groups.last().unwrap().label, "Gobo");
        assert_eq!(&*params.last().unwrap().name, "goboOffset");
        let (_, area) = d.schema(standard::AREA_LIGHT).unwrap();
        assert!(area.iter().all(|p| !p.name.starts_with("gobo")));
    }

    #[test]
    fn tints_and_flats_round_trip() {
        let red = material::network([0.7, 0.1, 0.05], "");
        assert_eq!(tint(&red), Some(Tint::Color([0.7, 0.1, 0.05])));
        assert_eq!(flat(&Tint::Color([0.7, 0.1, 0.05])), red);
        let leaves = Tint::Modulated {
            color: [0.2, 0.5, 0.1],
            attribute: "Cd".into(),
        };
        assert_eq!(tint(&flat(&leaves)), Some(leaves));
        assert_eq!(
            tint(&flat(&Tint::Attribute("Cd".into()))),
            Some(Tint::Modulated {
                color: [1.0; 3],
                attribute: "Cd".into()
            })
        );
        assert_eq!(tint(&default_material()), Some(Tint::Color([0.8; 3])));
        // A translucent leaf shows its color elsewhere; flat, it is opaque.
        let leaf = material::network_of(&PixelSurface {
            color: [0.2, 0.5, 0.1],
            translucency: 0.7,
            ..PixelSurface::default()
        });
        let shown = tint(&leaf).unwrap();
        assert_eq!(shown, Tint::Color([0.2, 0.5, 0.1]));
        assert_eq!(material::surface(&flat(&shown)).unwrap().translucency, 0.0);
        let surface = preview_surface(&red);
        assert_eq!(
            tint(&from_preview_surface(&surface)).and_then(|t| t.color()),
            Some([0.7, 0.1, 0.05])
        );
    }

    #[test]
    fn volumes_say_so_and_tint_by_their_scatter_color() {
        let fog = material::volume::network(&crate::PixelVolume {
            scatter_color: [0.6, 0.7, 0.8],
            ..crate::PixelVolume::default()
        });
        let d = description();
        assert!(d.volume(&fog));
        assert!(!d.volume(&default_material()));
        assert_eq!(tint(&fog), Some(Tint::Color([0.6, 0.7, 0.8])));
    }

    #[test]
    fn another_engines_material_renders_flat_in_its_tint() {
        // Spectre's tint is its base color: a stand-in owner with that rule.
        struct Other;
        impl MaterialRules for Other {
            fn tint(&self, _: &MaterialNetwork) -> Option<Tint> {
                Some(Tint::Color([0.2, 0.4, 0.6]))
            }
            fn volume(&self, _: &MaterialNetwork) -> bool {
                false
            }
            fn flat(&self, tint: &Tint) -> MaterialNetwork {
                flat(tint)
            }
            fn default_material(&self) -> MaterialNetwork {
                default_material()
            }
            fn preview_surface(&self, network: &MaterialNetwork) -> PreviewSurface {
                preview_surface(network)
            }
            fn import_preview_surface(&self, surface: &PreviewSurface) -> MaterialNetwork {
                from_preview_surface(surface)
            }
        }
        let owner = EngineDescription {
            id: Arc::from("other"),
            materials: Arc::new(Other),
            ..description()
        };
        let theirs = MaterialNetwork {
            engine: "other".into(),
            ..MaterialNetwork::default()
        };
        let adopted = description().adopt(theirs, &owner);
        assert_eq!(
            material::surface(&adopted),
            Some(PixelSurface {
                color: [0.2, 0.4, 0.6],
                attribute: "".into(),
                ..PixelSurface::default()
            })
        );
    }
}
