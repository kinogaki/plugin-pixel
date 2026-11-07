//! Pixel's material nodes and what only its code knows of them.
//!
//! - **Pixel Surface** (`kinogaki.pixel.pixel_surface`): Color, times the
//!   attribute its Color Attribute names (`Cd`) where a mesh has it;
//!   Translucency makes it a thin surface: light reaching its back comes
//!   through × Translucency × Translucency Color × its color, and light
//!   passing it keeps × Translucency × Translucency Color.
//! - **Pixel Volume** (`kinogaki.pixel.pixel_volume`): fills its closed mesh
//!   with a participating medium instead of shading its surface: Density
//!   (extinction a unit of distance), Scatter Color (the single-scattering
//!   albedo), Anisotropy (Henyey–Greenstein's g), Edge Falloff, Shadows, an
//!   Emission section and a Noise section (value-noise fBm in world space).
//!
//! Pixel has no node graph: each is a material on its own, its values
//! constants, the network's shown node. Of MaterialX it renders Surface
//! Unlit, as an unlit surface: its emission color × emission, whatever lights
//! it, letting through transmission × transmission color. [`Rules`]
//! approximates its materials for the shaded viewport, gives a Pixel Surface
//! as an OpenPBR Surface, and makes one of another material's.
use crate::render::{Medium, Surface};
use kinogaki_render_api::material::mtlx;
use kinogaki_render_api::material::standard::open_pbr;
use kinogaki_render_api::value::{Number, NumberLook, Parameters, Value, ValueKind};
use kinogaki_render_api::{
    Input, Link, MaterialDisplay, MaterialNetwork, MaterialRules, NodeDef, Parameter, Section,
};
use kinogaki_studio_api::NodeTypeId;
use std::sync::Arc;

/// The Pixel Surface's id.
pub const SURFACE: NodeTypeId = NodeTypeId::declared("kinogaki.pixel.pixel_surface");
/// The Pixel Volume's id.
pub const VOLUME: NodeTypeId = NodeTypeId::declared("kinogaki.pixel.pixel_volume");
/// MaterialX's Surface Unlit, which Pixel renders as it is.
const UNLIT: &str = "surface_unlit";

/// The groups a node's values sit in.
const MAIN: &str = "main";
const EMISSION: &str = "emission";
const NOISE: &str = "noise";

/// A constant `name` of `kind` at `default` in `group`.
fn value(
    name: &str,
    label: &str,
    group: &str,
    kind: ValueKind,
    default: impl Into<Value>,
) -> Input {
    Input::Setting(Parameter::new(name, label, kind, default).group(group))
}

/// A slider over `min..=max` dragged over `soft`, with `decimals`.
fn slider(min: f64, max: f64, soft: (f64, f64)) -> ValueKind {
    ValueKind::Number(Number::plain().range(min, max).soft(soft.0, soft.1))
}

/// `input` with `tooltip`, shown `decimals` places when a number.
fn tip(input: Input, tooltip: &str, decimals: u8) -> Input {
    match input {
        Input::Setting(d) => {
            let look = NumberLook::slider().decimals(decimals).speed(0.01);
            Input::Setting(d.help(tooltip).look(look))
        }
        other => other,
    }
}

/// The Pixel Surface.
fn surface() -> NodeDef {
    let mut def = NodeDef::new(SURFACE, "Pixel Surface", "Surface")
        .input(value("color", "Color", MAIN, ValueKind::Color, [0.8; 3]))
        .input(tip(
            value(
                "attribute",
                "Color Attribute",
                MAIN,
                ValueKind::attribute(),
                "Cd",
            ),
            "Multiplies Color where the mesh has this attribute",
            0,
        ))
        .input(tip(
            value(
                "translucency",
                "Translucency",
                MAIN,
                slider(0.0, 1.0, (0.0, 1.0)),
                0.0,
            ),
            "The share of light behind a thin surface that shows through",
            2,
        ))
        .input(tip(
            value(
                "translucency_color",
                "Translucency Color",
                MAIN,
                ValueKind::Color,
                [1.0; 3],
            ),
            "Tints the light that shows through; white keeps the Color",
            0,
        ));
    def.aliases = ["pixel", "material", "surface", "flat", "toon", "sprite"]
        .map(Arc::from)
        .to_vec();
    def.groups = vec![Section::new(MAIN, "Surface", true)];
    def
}

/// The Pixel Volume.
fn volume() -> NodeDef {
    let big = |soft| slider(0.0, 1e6, soft);
    let mut def = NodeDef::new(VOLUME, "Pixel Volume", "Surface")
        .input(tip(
            value("density", "Density", MAIN, big((0.0, 4.0)), 0.5),
            "The share of light a unit of distance takes out",
            3,
        ))
        .input(tip(
            value(
                "scatter_color",
                "Scatter Color",
                MAIN,
                ValueKind::Color,
                [0.8; 3],
            ),
            "The share of the light taken out that scatters; the rest is absorbed",
            0,
        ))
        .input(tip(
            value(
                "anisotropy",
                "Anisotropy",
                MAIN,
                slider(-0.9, 0.9, (-0.9, 0.9)),
                0.2,
            ),
            "Above 0 light scatters forward, brightest looking toward the light",
            2,
        ))
        .input(tip(
            value("edge_falloff", "Edge Falloff", MAIN, big((0.0, 10.0)), 0.0),
            "How far in from the bounds the density fades in",
            2,
        ))
        .input(tip(
            value("shadows", "Shadows", MAIN, ValueKind::Toggle, false),
            "Dims the light passing through it, on surfaces and on itself",
            0,
        ))
        .input(tip(
            value("emission", "Emission", EMISSION, big((0.0, 4.0)), 0.0),
            "Light it gives off",
            3,
        ))
        .input(value(
            "emission_color",
            "Emission Color",
            EMISSION,
            ValueKind::Color,
            [1.0; 3],
        ))
        .input(tip(
            value(
                "noise_amount",
                "Amount",
                NOISE,
                slider(0.0, 1.0, (0.0, 1.0)),
                0.0,
            ),
            "How much noise shapes the density",
            3,
        ))
        .input(tip(
            value("noise_frequency", "Frequency", NOISE, big((0.0, 4.0)), 0.5),
            "Noise cycles a scene unit",
            3,
        ))
        .input(value(
            "noise_octaves",
            "Octaves",
            NOISE,
            ValueKind::integer(1, 8),
            3i64,
        ))
        .input(value(
            "noise_offset",
            "Offset",
            NOISE,
            ValueKind::Vector {
                components: 3,
                number: Number::length(),
            },
            Value::Numbers(vec![0.0; 3]),
        ))
        .input(value(
            "noise_seed",
            "Seed",
            NOISE,
            ValueKind::integer(0, 9999),
            0i64,
        ));
    def.aliases = [
        "volume", "fog", "mist", "smoke", "haze", "god rays", "medium",
    ]
    .map(Arc::from)
    .to_vec();
    def.groups = vec![
        Section::new(MAIN, "Volume", true),
        Section::new(EMISSION, "Emission", false),
        Section::new(NOISE, "Noise", false),
    ];
    def
}

/// Pixel's nodes, in the order a host lists them.
pub fn defs() -> Vec<NodeDef> {
    vec![surface(), volume()]
}

/// What a material renders as.
#[derive(Clone, Debug, PartialEq)]
pub enum Material {
    /// A surface, its color times the attribute named (none: its color alone).
    Surface(Surface, Option<Arc<str>>),
    /// A medium filling its mesh.
    Volume(Medium),
}

impl Default for Material {
    /// The default surface, reading no attribute.
    fn default() -> Material {
        Material::Surface(Surface::default(), None)
    }
}

/// Reads constants of a node's values.
struct Values<'a>(&'a Parameters);

impl Values<'_> {
    fn number(&self, name: &str, default: f32) -> f32 {
        let v = self.0.get(name).and_then(Value::number).map(|v| v as f32);
        v.filter(|v| v.is_finite()).unwrap_or(default)
    }
    fn color(&self, name: &str, default: [f32; 3]) -> [f32; 3] {
        let v = self.0.get(name).and_then(Value::color);
        v.map_or(default, |c| {
            c.map(|c| {
                if c.is_finite() {
                    c.max(0.0) as f32
                } else {
                    0.0
                }
            })
        })
    }
    fn integer(&self, name: &str, default: i64) -> i64 {
        self.0.get(name).and_then(Value::integer).unwrap_or(default)
    }
}

/// What `network` renders as: its surface's material (a Pixel Surface or
/// Volume, or MaterialX's Surface Unlit; anything else, or none: the
/// default).
pub fn material(network: &MaterialNetwork) -> Material {
    let Some(node) = network.surface().map(|at| &network.nodes()[at as usize]) else {
        return Material::default();
    };
    let v = Values(&node.values);
    if node.def.mtlx().is_some_and(|d| d.node == UNLIT) {
        let emission = v.number("emission", 1.0).max(0.0);
        let surface = Surface {
            color: v.color("emission_color", [1.0; 3]).map(|c| c * emission),
            translucency: v.number("transmission", 0.0).clamp(0.0, 1.0),
            translucency_color: v.color("transmission_color", [1.0; 3]),
            unlit: true,
        };
        return Material::Surface(surface, None);
    }
    if node.def.id == VOLUME {
        let offset = node
            .values
            .get("noise_offset")
            .and_then(Value::numbers)
            .unwrap_or(&[]);
        return Material::Volume(Medium {
            density: v.number("density", 0.5).max(0.0),
            scatter_color: v.color("scatter_color", [0.8; 3]).map(|c| c.min(1.0)),
            anisotropy: v.number("anisotropy", 0.2).clamp(-0.9, 0.9),
            emission: v
                .color("emission_color", [1.0; 3])
                .map(|c| c * v.number("emission", 0.0).max(0.0)),
            noise_amount: v.number("noise_amount", 0.0).clamp(0.0, 1.0),
            noise_frequency: v.number("noise_frequency", 0.5).max(0.0),
            noise_octaves: v.integer("noise_octaves", 3).clamp(1, 8) as u32,
            noise_offset: std::array::from_fn(|k| offset.get(k).copied().unwrap_or(0.0) as f32),
            noise_seed: v.integer("noise_seed", 0).clamp(0, i64::from(u32::MAX)) as u32,
            edge_falloff: v.number("edge_falloff", 0.0).max(0.0),
            shadows: node
                .values
                .get("shadows")
                .and_then(Value::bool)
                .unwrap_or(false),
        });
    }
    if node.def.id != SURFACE {
        return Material::default();
    }
    let attribute = (node.values.get("attribute").and_then(Value::text))
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .map(Arc::from);
    let surface = Surface {
        color: v.color("color", [0.8; 3]),
        translucency: v.number("translucency", 0.0).clamp(0.0, 1.0),
        translucency_color: v.color("translucency_color", [1.0; 3]),
        unlit: false,
    };
    Material::Surface(surface, attribute)
}

/// The Pixel Surface node holding `values`, alone in a network.
fn network_of(values: Parameters) -> Option<MaterialNetwork> {
    let def = Arc::new(surface());
    let mut built = MaterialNetwork::builder();
    let at = built.node(&def, &values).ok()?;
    built.build(at).ok()
}

/// Pixel's material rules.
pub struct Rules;

impl MaterialRules for Rules {
    /// A surface's color, times the attribute it names; a volume's scatter
    /// color, filling its shape.
    fn display(&self, network: &MaterialNetwork) -> MaterialDisplay {
        match material(network) {
            Material::Surface(s, attribute) => MaterialDisplay {
                primvar: attribute,
                ..MaterialDisplay::color(s.color)
            },
            Material::Volume(m) => MaterialDisplay {
                volume: true,
                ..MaterialDisplay::color(m.scatter_color)
            },
        }
    }

    /// A Pixel Surface as an OpenPBR Surface of its color; a Pixel Volume
    /// has none.
    fn standard(&self, own: &MaterialNetwork) -> Option<MaterialNetwork> {
        let Material::Surface(surface, _) = material(own) else {
            return None;
        };
        let color = Value::Color(surface.color.map(f64::from));
        open_pbr(&[("base_color".to_owned(), color)].into()).ok()
    }

    /// The standard form, an OpenPBR Surface, as Pixel renders it: light
    /// alone (no base, no specular) a Surface Unlit of its emission; else a
    /// Pixel Surface of its base color, or of white times the attribute a
    /// geometric color or property drives the base color with.
    fn translate(&self, standard: &MaterialNetwork) -> Option<MaterialNetwork> {
        let at = standard.surface()?;
        let shown = &standard.nodes()[at as usize];
        let number = |name: &str| shown.values.get(name).and_then(Value::number);
        let color = |name: &str| shown.values.get(name).and_then(Value::color);
        if number("base_weight") == Some(0.0) && number("specular_weight") == Some(0.0) {
            let values = [
                ("emission", Value::Number(number("emission_luminance")?)),
                ("emission_color", Value::Color(color("emission_color")?)),
            ];
            let unlit = mtlx::def("ND_surface_unlit")?;
            let values = values.map(|(n, v)| (n.to_owned(), v)).into();
            let mut built = MaterialNetwork::builder();
            let at = built.node(&unlit, &values).ok()?;
            return built.build(at).ok();
        }
        let into = |l: &&Link| l.to == at && &*l.input == "base_color";
        let read = (standard.links().iter().find(into)).and_then(|l| {
            let node = &standard.nodes()[l.from as usize];
            match node.def.mtlx()?.node.as_str() {
                "geomcolor" => Some("Cd"),
                "geompropvalue" => node.values.get("geomprop")?.text(),
                _ => None,
            }
        });
        let base = match read {
            Some(_) => [1.0; 3],
            None => color("base_color")?,
        };
        let mut values: Parameters = [("color".to_owned(), Value::Color(base))].into();
        if let Some(name) = read {
            values.insert("attribute".to_owned(), Value::Text(name.to_owned()));
        }
        network_of(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// OpenPBR Surface's signature.
    const OPEN_PBR: &str = "ND_open_pbr_surface_surfaceshader";

    fn of(def: NodeDef, values: &[(&str, Value)]) -> MaterialNetwork {
        let values: Parameters = values
            .iter()
            .map(|(n, v)| ((*n).to_owned(), v.clone()))
            .collect();
        let mut built = MaterialNetwork::builder();
        let at = built.node(&Arc::new(def), &values).unwrap();
        built.build(at).unwrap()
    }

    #[test]
    fn a_surface_reads_its_constants_and_its_attribute() {
        let leaf = of(
            surface(),
            &[
                ("color", Value::Color([0.2, 0.5, 0.1])),
                ("translucency", Value::Number(0.5)),
                ("translucency_color", Value::Color([1.0, 0.5, 0.0])),
            ],
        );
        let Material::Surface(s, attribute) = material(&leaf) else {
            panic!("a surface");
        };
        assert_eq!(
            (s.color, s.transmittance()),
            ([0.2, 0.5, 0.1], [0.5, 0.25, 0.0])
        );
        assert_eq!(attribute.as_deref(), Some("Cd"), "the default attribute");
        assert_eq!(
            leaf.primvars().iter().map(|p| &**p).collect::<Vec<_>>(),
            ["Cd"]
        );
        let shown = Rules.display(&leaf);
        assert_eq!(
            (shown.color, shown.primvar.as_deref(), shown.volume),
            ([0.2, 0.5, 0.1], Some("Cd"), false)
        );
        assert_eq!(material(&MaterialNetwork::default()), Material::default());
    }

    #[test]
    fn a_volume_reads_its_medium_and_shows_as_one() {
        let mist = of(
            volume(),
            &[
                ("density", Value::Number(0.15)),
                ("emission", Value::Number(2.0)),
                ("emission_color", Value::Color([1.0, 0.5, 0.0])),
                ("noise_octaves", Value::Integer(5)),
            ],
        );
        let Material::Volume(m) = material(&mist) else {
            panic!("a volume");
        };
        assert_eq!(
            (m.density, m.emission, m.noise_octaves),
            (0.15, [2.0, 1.0, 0.0], 5)
        );
        let shown = Rules.display(&mist);
        assert!(shown.volume && shown.color == [0.8; 3]);
    }

    /// `values` by name.
    fn values(values: &[(&str, Value)]) -> Parameters {
        (values.iter())
            .map(|(n, v)| ((*n).to_owned(), v.clone()))
            .collect()
    }

    /// A node feeding a surface: its signature, its values and the input
    /// it feeds.
    type Fed<'a> = (&'a str, &'a [(&'a str, Value)], &'a str);

    /// MaterialX node `signature` holding `held`, shown by a Surface
    /// Material, `fed` wired into it.
    fn shown(signature: &str, held: &[(&str, Value)], fed: Option<Fed<'_>>) -> MaterialNetwork {
        let mut built = MaterialNetwork::builder();
        let s = (built.node(&mtlx::def(signature).unwrap(), &values(held))).unwrap();
        if let Some((by, held, into)) = fed {
            let n = built.node(&mtlx::def(by).unwrap(), &values(held)).unwrap();
            built.link(n, "out", s, into).unwrap();
        }
        let m = built
            .node(
                &mtlx::def("ND_surfacematerial").unwrap(),
                &Parameters::new(),
            )
            .unwrap();
        built.link(s, "out", m, "surfaceshader").unwrap();
        built.build(m).unwrap()
    }

    #[test]
    fn surface_unlit_is_an_unlit_surface_of_its_emission() {
        let lamp = shown(
            "ND_surface_unlit",
            &[
                ("emission", Value::Number(2.0)),
                ("emission_color", Value::Color([0.5, 0.25, 1.0])),
                ("transmission", Value::Number(0.5)),
            ],
            None,
        );
        let Material::Surface(s, attribute) = material(&lamp) else {
            panic!("a surface");
        };
        assert!(s.unlit && attribute.is_none());
        assert_eq!((s.color, s.transmittance()), ([1.0, 0.5, 2.0], [0.5; 3]));
    }

    #[test]
    fn the_standard_form_of_light_alone_is_unlit_and_an_attribute_drives_the_color() {
        let glow = [
            ("base_weight", Value::Number(0.0)),
            ("specular_weight", Value::Number(0.0)),
            ("emission_luminance", Value::Number(1.0)),
            ("emission_color", Value::Color([0.9, 0.3, 0.1])),
        ];
        let glow = Rules.translate(&shown(OPEN_PBR, &glow, None)).unwrap();
        let Material::Surface(s, _) = material(&glow) else {
            panic!("a surface");
        };
        assert!(s.unlit);
        assert_eq!(s.color, [0.9, 0.3, 0.1]);
        let paint = [("geomprop", Value::Text("paint".to_owned()))];
        let paint = Some(("ND_geompropvalue_color3", &paint[..], "base_color"));
        let painted = shown(OPEN_PBR, &[], paint);
        let Material::Surface(s, attribute) = material(&Rules.translate(&painted).unwrap()) else {
            panic!("a surface");
        };
        assert_eq!((s.color, attribute.as_deref()), ([1.0; 3], Some("paint")));
    }

    #[test]
    fn materialx_reaches_pixel_as_surface_unlit_or_its_standard_forms_surface() {
        use kinogaki_render_api::Engines;
        let pixel = crate::description::description();
        let engines = Engines::new(vec![pixel.clone()]);
        // glTF PBR: a Pixel Surface of its standard form's base color.
        let orange = [("base_color", Value::Color([0.9, 0.5, 0.1]))];
        let gltf = shown("ND_gltf_pbr_surfaceshader", &orange, None);

        let Material::Surface(s, _) = material(&engines.adopt(&gltf, pixel.id())) else {
            panic!("a surface");
        };
        assert_eq!((s.color, s.unlit), ([0.9, 0.5, 0.1], false));
        // USD Preview Surface: lit, its diffuse color, its emission and
        // cutout aside (a Pixel Surface has neither).
        let leaf = [
            ("diffuseColor", Value::Color([0.2, 0.6, 0.3])),
            ("emissiveColor", Value::Color([0.5, 0.5, 0.5])),
            ("opacityThreshold", Value::Number(0.5)),
        ];
        let usd = shown("ND_UsdPreviewSurface_surfaceshader", &leaf, None);
        let Material::Surface(s, _) = material(&engines.adopt(&usd, pixel.id())) else {
            panic!("a surface");
        };
        assert_eq!((s.color, s.unlit), ([0.2, 0.6, 0.3], false));
        // Surface Unlit of an image: as it is, the image cut and named.
        let image = Some(("ND_image_color3", &[][..], "emission_color"));
        let lamp = engines.adopt(&shown("ND_surface_unlit", &[], image), pixel.id());
        assert!(matches!(material(&lamp), Material::Surface(s, _) if s.unlit));
        assert_eq!(lamp.cut(), [std::sync::Arc::from("image")]);
    }

    #[test]
    fn a_surface_is_an_openpbr_surface_of_its_color_and_a_volume_none() {
        let color = Value::Color([0.25, 0.5, 0.125]);
        let mine = of(surface(), &[("color", color.clone())]);
        let standard = Rules.standard(&mine).unwrap();
        let shown = &standard.nodes()[standard.surface().unwrap() as usize];
        assert_eq!(shown.def.mtlx().unwrap().node, "open_pbr_surface");
        assert_eq!(shown.values["base_color"], color);
        assert!(Rules.standard(&of(volume(), &[])).is_none());
    }

    #[test]
    fn the_standard_form_becomes_a_surface_of_its_base_color() {
        let color = [("base_color".to_owned(), Value::Color([0.1, 0.2, 0.3]))];
        let standard = kinogaki_render_api::material::standard::open_pbr(&color.into()).unwrap();
        let mine = Rules.translate(&standard).unwrap();
        let Material::Surface(s, _) = material(&mine) else {
            panic!("a surface");
        };
        assert_eq!(s.color, [0.1, 0.2, 0.3]);
    }
}
