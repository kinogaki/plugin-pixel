//! Pixel's material nodes and what only its code knows of them.
//!
//! - **Pixel Surface** (`kinogaki.pixel.pixel_surface`): Color (a texture
//!   node may drive it: its picture times its factor), times the attribute
//!   its Color Attribute names (`Cd`) where a mesh has it; Cutout cuts it
//!   out where the texture's alpha (times glTF's vertex alpha) falls below
//!   it; Translucency makes it a
//!   thin surface: light reaching its back comes through × Translucency ×
//!   Translucency Color × its color, and light passing it keeps ×
//!   Translucency × Translucency Color.
//! - **Pixel Volume** (`kinogaki.pixel.pixel_volume`): fills its closed mesh
//!   with a participating medium instead of shading its surface: Density
//!   (extinction a unit of distance), Scatter Color (the single-scattering
//!   albedo), Anisotropy (Henyey–Greenstein's g), Edge Falloff, Shadows, an
//!   Emission section and a Noise section (value-noise fBm in world space).
//!
//! Pixel has no node graph: each is a material on its own, its values
//! constants, the network's shown node, but for the texture node driving its
//! color (MaterialX's `image` and glTF's texture nodes, read natively:
//! [`sampled`]). Of MaterialX it renders Surface Unlit (and Kinogaki's
//! Surface Unlit Cutout), as an unlit surface: its emission color ×
//! emission, whatever lights it, letting through transmission ×
//! transmission color. [`Rules`] approximates its materials for the shaded
//! viewport, gives a Pixel Surface as an OpenPBR Surface, and makes one of
//! another material's, its base color's texture and alpha cutout kept.
use crate::render::{Medium, Surface, Wrap};
use crate::sampled::{Sampled, is_texture, sampled};
use kinogaki_render_api::geometry::image::{self, Sampling};
use kinogaki_render_api::geometry::names;
use kinogaki_render_api::material::mtlx;
use kinogaki_render_api::materialx::Type;
use kinogaki_render_api::value::{Number, NumberLook, Parameters, Value, ValueKind};
use kinogaki_render_api::{
    DisplayTexture, MaterialDisplay, MaterialNetwork, MaterialRules, NetworkBuilder, NodeDef,
    NodeIndex, NodeInput, Parameter, Section,
};
use kinogaki_studio_api::NodeTypeId;
use std::collections::BTreeMap;
use std::sync::Arc;

/// The Pixel Surface's id.
pub const SURFACE: NodeTypeId = NodeTypeId::declared("kinogaki.pixel.pixel_surface");
/// The Pixel Volume's id.
pub const VOLUME: NodeTypeId = NodeTypeId::declared("kinogaki.pixel.pixel_volume");
/// MaterialX's Surface Unlit, and Kinogaki's Cutout of it, which Pixel
/// renders as they are.
const UNLIT: [&str; 2] = ["surface_unlit", "kinogaki_surface_unlit"];

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
) -> NodeInput {
    NodeInput::Setting(Parameter::new(name, label, kind, default).group(group))
}

/// A slider over `min..=max` dragged over `soft`, with `decimals`.
fn slider(min: f64, max: f64, soft: (f64, f64)) -> ValueKind {
    ValueKind::Number(Number::plain().range(min, max).soft(soft.0, soft.1))
}

/// `input` with `tooltip`, shown `decimals` places when a number.
fn tip(input: NodeInput, tooltip: &str, decimals: u8) -> NodeInput {
    match input {
        NodeInput::Setting(d) => {
            let look = NumberLook::slider().decimals(decimals).speed(0.01);
            NodeInput::Setting(d.help(tooltip).look(look))
        }
        other => other,
    }
}

/// The Pixel Surface.
fn surface() -> NodeDef {
    let color = Parameter::new("color", "Color", ValueKind::Color, [0.8; 3]).group(MAIN);
    let mut def = NodeDef::new(SURFACE, "Pixel Surface", "Surface")
        .input(NodeInput::Driven(color, Type::Color3))
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
        ))
        .input(tip(
            value("cutout", "Cutout", MAIN, slider(0.0, 1.0, (0.0, 1.0)), 0.0),
            "Cuts the surface out where its color's texture is less opaque than this",
            2,
        ));
    def.aliases = ["pixel", "material", "surface", "flat", "toon", "sprite"]
        .map(Arc::from)
        .to_vec();
    def.section(Section::new(MAIN, "Surface", true))
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
    def.section(Section::new(MAIN, "Volume", true))
        .section(Section::new(EMISSION, "Emission", false))
        .section(Section::new(NOISE, "Noise", false))
}

/// Pixel's nodes, in the order a host lists them.
pub fn defs() -> Vec<NodeDef> {
    vec![surface(), volume()]
}

/// What a material renders as.
#[derive(Clone, Debug, PartialEq)]
pub enum Material {
    /// A surface, its color times what it reads of its mesh and images.
    Surface(Surface, Look),
    /// A medium filling its mesh.
    Volume(Medium),
}

impl Default for Material {
    /// The default surface, reading nothing.
    fn default() -> Material {
        Material::Surface(Surface::default(), Look::default())
    }
}

/// What a surface's color is multiplied by beside its constants.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Look {
    /// The attribute tinting each corner (none: white).
    pub attribute: Option<Arc<str>>,
    /// The texture it samples.
    pub texture: Option<Sampled>,
    /// The alpha below which the texture cuts the surface out.
    pub cutout: Option<f32>,
}

impl Look {
    /// The attribute tinting each corner: the surface's own, else its
    /// texture's (glTF's vertex colors).
    pub fn tint(&self) -> Option<&Arc<str>> {
        (self.attribute.as_ref()).or_else(|| self.texture.as_ref()?.tint.as_ref())
    }

    /// The attribute multiplying its texture's alpha at each corner (glTF's
    /// vertex alpha).
    pub fn alpha(&self) -> Option<&Arc<str>> {
        self.texture.as_ref()?.alpha.as_ref()
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
    let Some(at) = network.surface() else {
        return Material::default();
    };
    let node = &network.nodes()[at as usize];
    let v = Values(node.values());
    if let Some(unlit) = node
        .def()
        .mtlx()
        .filter(|d| UNLIT.contains(&d.node.as_str()))
    {
        let emission = v.number("emission", 1.0).max(0.0);
        let surface = Surface {
            color: v.color("emission_color", [1.0; 3]).map(|c| c * emission),
            translucency: v.number("transmission", 0.0).clamp(0.0, 1.0),
            translucency_color: v.color("transmission_color", [1.0; 3]),
            unlit: true,
            texture: None,
        };
        // Surface Unlit Cutout's mask mode cuts out at its cutoff.
        let masks = unlit.node != UNLIT[0] && v.integer("alpha_mode", 2) == 1;
        let mut look = read(network, at, "emission_color");
        look.cutout = (masks && look.texture.is_some()).then(|| v.number("alpha_cutoff", 0.5));
        return Material::Surface(surface, look);
    }
    if node.def().id == VOLUME {
        let offset = node
            .values()
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
                .values()
                .get("shadows")
                .and_then(Value::bool)
                .unwrap_or(false),
        });
    }
    if node.def().id != SURFACE {
        return Material::default();
    }
    let attribute = (node.values().get("attribute").and_then(Value::text))
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .map(Arc::from);
    let surface = Surface {
        color: v.color("color", [0.8; 3]),
        translucency: v.number("translucency", 0.0).clamp(0.0, 1.0),
        translucency_color: v.color("translucency_color", [1.0; 3]),
        unlit: false,
        texture: None,
    };
    let mut look = read(network, at, "color");
    let cutout = v.number("cutout", 0.0).clamp(0.0, 1.0);
    look.cutout = (cutout > 0.0 && look.texture.is_some()).then_some(cutout);
    look.attribute = attribute.or(look.attribute);
    Material::Surface(surface, look)
}

/// What drives input `input` of node `at`: a texture node, read; a
/// geometric color or property, its attribute.
fn read(network: &MaterialNetwork, at: NodeIndex, input: &str) -> Look {
    let Some(from) = driver(network, at, input) else {
        return Look::default();
    };
    let node = &network.nodes()[from as usize];
    let attribute = match node.def().mtlx().map(|d| d.node.as_str()) {
        Some("geomcolor") => Some(Arc::from(names::CD)),
        Some("geompropvalue") => node
            .values()
            .get("geomprop")
            .and_then(Value::text)
            .map(Arc::from),
        _ => None,
    };
    Look {
        attribute,
        texture: sampled(network, from),
        cutout: None,
    }
}

/// What drives input `input` of node `at`.
fn driver(network: &MaterialNetwork, at: NodeIndex, input: &str) -> Option<NodeIndex> {
    (network.links().iter())
        .find(|l| l.to == at && &*l.input == input)
        .map(|l| l.from)
}

/// Node `at` of `from` and every node driving it, into `built`, each once
/// (`made` maps them): its index there.
fn copied(
    from: &MaterialNetwork,
    at: NodeIndex,
    built: &mut NetworkBuilder,
    made: &mut BTreeMap<NodeIndex, NodeIndex>,
) -> Option<NodeIndex> {
    if let Some(&there) = made.get(&at) {
        return Some(there);
    }
    let node = &from.nodes()[at as usize];
    let there = built.node(node.def(), node.values()).ok()?;
    for (input, image) in node.images() {
        built.image(there, input, image.clone()).ok()?;
    }
    made.insert(at, there);
    for l in from.links().iter().filter(|l| l.to == at) {
        let source = copied(from, l.from, built, made)?;
        built.link(source, &l.output, there, &l.input).ok()?;
    }
    Some(there)
}

/// Texture node `at` of `from` (and what drives it) copied into `built`,
/// its color driving input `input` of node `to`.
fn drive(
    built: &mut NetworkBuilder,
    (from, at): (&MaterialNetwork, NodeIndex),
    (to, input): (NodeIndex, &str),
) -> Option<()> {
    let source = copied(from, at, built, &mut BTreeMap::new())?;
    built.link(source, output(from, at), to, input).ok()
}

/// Pixel's material rules.
pub struct Rules;

impl MaterialRules for Rules {
    /// A surface's color, times the attribute it names and its texture; a
    /// volume's scatter color, filling its shape.
    fn display(&self, network: &MaterialNetwork) -> MaterialDisplay {
        match material(network) {
            Material::Surface(s, look) => {
                let texture = (look.texture.as_ref()).and_then(|t| {
                    let mut shown = DisplayTexture::new(t.source.clone()?);
                    shown.uv = t.uv.clone();
                    shown.transform = t.transform;
                    shown.sampling = Sampling {
                        wrap: t.wrap.map(|w| match w {
                            Wrap::Repeat => image::Wrap::Repeat,
                            Wrap::Clamp => image::Wrap::Clamp,
                            Wrap::Mirror => image::Wrap::MirrorRepeat,
                            Wrap::Black => image::Wrap::Black,
                        }),
                        filter: match t.nearest {
                            true => image::SampleFilter::Nearest,
                            false => image::SampleFilter::Bilinear,
                        },
                    };
                    Some(shown)
                });
                let factor = look.texture.as_ref().map_or([1.0; 4], |t| t.factor);
                MaterialDisplay {
                    primvar: look.tint().cloned(),
                    texture,
                    ..MaterialDisplay::color([0, 1, 2].map(|k| s.color[k] * factor[k]))
                }
            }
            Material::Volume(m) => MaterialDisplay {
                volume: true,
                ..MaterialDisplay::color(m.scatter_color)
            },
        }
    }

    /// A Pixel Surface as an OpenPBR Surface of its color, its texture
    /// driving the base color; a Pixel Volume has none.
    fn standard(&self, own: &MaterialNetwork) -> Option<MaterialNetwork> {
        let Material::Surface(surface, _) = material(own) else {
            return None;
        };
        let color = Value::Color(surface.color.map(f64::from));
        let values = [("base_color".to_owned(), color)].into();
        let mut built = MaterialNetwork::builder();
        let s = built
            .node(&mtlx::def("ND_open_pbr_surface_surfaceshader")?, &values)
            .ok()?;
        let shown = own.surface()?;
        if let Some(t) = driver(own, shown, "color").filter(|&t| is_texture(own, t)) {
            drive(&mut built, (own, t), (s, "base_color"))?;
        }
        // Shown through a Surface Material, as every standard form is.
        let m = built
            .node(&mtlx::def("ND_surfacematerial")?, &Parameters::new())
            .ok()?;
        built.link(s, "out", m, "surfaceshader").ok()?;
        built.build(m).ok()
    }

    /// The standard form, an OpenPBR Surface, as Pixel renders it: light
    /// alone (no base, no specular) a Surface Unlit of its emission; else a
    /// Pixel Surface of its base color, of the texture node driving it (and
    /// cut out at the cutoff its opacity is compared with), or of white
    /// times the attribute a geometric color or property drives it with.
    fn translate(&self, standard: &MaterialNetwork) -> Option<MaterialNetwork> {
        let at = standard.surface()?;
        let shown = &standard.nodes()[at as usize];
        let number = |name: &str| shown.values().get(name).and_then(Value::number);
        let color = |name: &str| shown.values().get(name).and_then(Value::color);
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
        let base = driver(standard, at, "base_color");
        let read = base.and_then(|b| {
            let node = &standard.nodes()[b as usize];
            match node.def().mtlx()?.node.as_str() {
                "geomcolor" => Some("Cd"),
                "geompropvalue" => node.values().get("geomprop")?.text(),
                _ => None,
            }
        });
        let textured = base.filter(|&b| is_texture(standard, b));
        let white = read.is_some() || textured.is_some();
        let mut values: Parameters = match white {
            true => [("color".to_owned(), Value::Color([1.0; 3]))].into(),
            false => [("color".to_owned(), Value::Color(color("base_color")?))].into(),
        };
        if let Some(name) = read {
            values.insert("attribute".to_owned(), Value::Text(name.to_owned()));
        }
        // glTF's mask: the opacity a cutoff compares the alpha with.
        let cutoff = driver(standard, at, "geometry_opacity").and_then(|o| {
            let node = &standard.nodes()[o as usize];
            let compares = node.def().mtlx()?.node == "ifgreatereq";
            let alpha = driver(standard, o, "value1").is_some();
            (compares && alpha).then(|| node.values().get("value2")?.number())?
        });
        if let (Some(cutoff), Some(_)) = (cutoff, textured) {
            values.insert("cutout".to_owned(), Value::Number(cutoff));
        }
        let mut built = MaterialNetwork::builder();
        let shown = built.node(&Arc::new(surface()), &values).ok()?;
        if let Some(t) = textured {
            drive(&mut built, (standard, t), (shown, "color"))?;
        }
        built.build(shown).ok()
    }
}

/// The output of texture node `at` that gives its color: `outcolor` of a
/// glTF color image, else `out`.
fn output(network: &MaterialNetwork, at: NodeIndex) -> &'static str {
    let def = network.nodes()[at as usize].def().mtlx();
    match def.is_some_and(|d| d.node.ends_with("colorimage")) {
        true => "outcolor",
        false => "out",
    }
}

#[cfg(test)]
#[path = "nodes_tests.rs"]
mod tests;
