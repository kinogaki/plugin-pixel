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
//! constants, the network's shown node. [`Rules`] approximates them for the
//! shaded viewport and makes a Pixel Surface of another engine's material.
use crate::render::{Medium, Surface};
use kinogaki_render_api::value::{Number, NumberLook, Parameters, Value, ValueKind};
use kinogaki_render_api::{
    Input, MaterialDisplay, MaterialNetwork, MaterialRules, NodeDef, Parameter, Section,
};
use kinogaki_studio_api::NodeTypeId;
use std::sync::Arc;

/// The Pixel Surface's id.
pub const SURFACE: NodeTypeId = NodeTypeId::declared("kinogaki.pixel.pixel_surface");
/// The Pixel Volume's id.
pub const VOLUME: NodeTypeId = NodeTypeId::declared("kinogaki.pixel.pixel_volume");

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

/// The node `network` shows when Pixel doesn't declare it (it renders the
/// default surface).
pub fn foreign(network: &MaterialNetwork) -> Option<NodeTypeId> {
    let node = network.output()?;
    (node.def != SURFACE && node.def != VOLUME).then_some(node.def)
}

/// What `network` renders as: its shown node's material (a node it doesn't
/// declare, or none: the default).
pub fn material(network: &MaterialNetwork) -> Material {
    let Some(node) = network.output() else {
        return Material::default();
    };
    let v = Values(&node.values);
    if node.def == VOLUME {
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
    if node.def != SURFACE {
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

    /// Another engine's material as a Pixel Surface of the color its shown
    /// node holds, under the names color usually goes by.
    fn translate(&self, foreign: &MaterialNetwork) -> Option<MaterialNetwork> {
        let shown = foreign.output()?;
        let names = ["color", "base_color", "diffuse_color", "diffuseColor"];
        let color = names.iter().find_map(|n| shown.values.get(*n)?.color())?;
        network_of([("color".to_owned(), Value::Color(color))].into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn another_engines_material_becomes_a_surface_of_its_color() {
        let mut other = surface();
        other.id = NodeTypeId::declared("acme.ray.plastic");
        let theirs = of(other, &[("color", Value::Color([0.1, 0.2, 0.3]))]);
        let mine = Rules.translate(&theirs).unwrap();
        let Material::Surface(s, _) = material(&mine) else {
            panic!("a surface");
        };
        assert_eq!(s.color, [0.1, 0.2, 0.3]);
    }
}
