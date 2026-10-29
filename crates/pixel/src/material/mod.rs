//! Pixel materials: the vocabulary, two terminals, and what a network of
//! them renders as ([`read`]): the Pixel Surface (Color, Translucency and
//! Translucency Color inputs and a Color Attribute setting) shades a mesh's
//! surface as a [`PixelSurface`]; the Pixel Volume ([`volume`]) fills a
//! closed mesh with a medium, a [`PixelVolume`](crate::PixelVolume),
//! and its surface is not drawn.
//!
//! A Pixel Surface's albedo is its Color times the mesh attribute its Color
//! Attribute names (`Cd` by default) where the mesh has that attribute, and
//! its Color alone where it does not. The setting is an Attribute param, so
//! a host sends meshes the attribute (`MaterialNetwork::attribute_names`).
//! Translucency makes it a thin surface: light reaching its back face comes
//! through, × Translucency × Translucency Color × the albedo, and light
//! passing it on the way to a surface behind keeps × Translucency ×
//! Translucency Color.

use std::sync::Arc;

use kinogaki_render_api::{
    Constant, MaterialNetwork, NodeLayout, NodeType, ParamKind, ParamSpec, ParamValue, Params,
    ShaderNode, SocketKind, SocketSpec,
};

use crate::scene::{PixelMaterial, PixelSurface};

pub mod volume;

pub use volume::VOLUME;

/// The engine's id: what its networks, settings and files name.
pub const ID: &str = "pixel";

/// The Pixel Surface's type name.
pub const SURFACE: &str = "pixel_surface";

/// Its Color input's name.
pub const COLOR: &str = "color";

/// Its Translucency input's name.
pub const TRANSLUCENCY: &str = "translucency";

/// Its Translucency Color input's name.
pub const TRANSLUCENCY_COLOR: &str = "translucency_color";

/// Its Color Attribute setting's name.
pub const ATTRIBUTE: &str = "attribute";

/// The attribute a new Pixel Surface reads: Houdini's color.
pub const DEFAULT_ATTRIBUTE: &str = "Cd";

/// The node types' layout: the Pixel Surface takes a color and its
/// translucency, names the attribute that multiplies the color, and is the
/// material's terminal (no outputs).
fn layout(_name: &str, _params: &Params) -> NodeLayout {
    let opaque = PixelSurface::default();
    NodeLayout {
        params: vec![ParamSpec {
            name: Arc::from(ATTRIBUTE),
            label: Arc::from("Color Attribute"),
            kind: ParamKind::Attribute,
            default: ParamValue::Attribute(Arc::from(DEFAULT_ATTRIBUTE)),
            tooltip: Some(Arc::from(
                "Multiplies Color where the mesh has this attribute",
            )),
            shown: true,
        }],
        inputs: vec![
            SocketSpec::new(COLOR, "Color", SocketKind::Color, opaque.color),
            SocketSpec {
                tooltip: Some(Arc::from(
                    "The share of light behind a thin surface that shows through",
                )),
                ..SocketSpec::new(
                    TRANSLUCENCY,
                    "Translucency",
                    SocketKind::Float,
                    [opaque.translucency; 3],
                )
            },
            SocketSpec {
                tooltip: Some(Arc::from(
                    "Tints the light that shows through; white keeps the Color",
                )),
                ..SocketSpec::new(
                    TRANSLUCENCY_COLOR,
                    "Translucency Color",
                    SocketKind::Color,
                    opaque.translucency_color,
                )
            },
        ],
        outputs: Vec::new(),
    }
}

/// Pixel Render's node types, by category: the Pixel Surface, then the
/// Pixel Volume.
pub fn node_types() -> Vec<Arc<NodeType>> {
    vec![Arc::new(surface_type()), Arc::new(volume::node_type())]
}

/// What `network` renders as: a volume when its terminal is a Pixel
/// Volume, else its Pixel Surface; none when it is not a Pixel network
/// with either.
pub fn read(network: &MaterialNetwork) -> Option<PixelMaterial> {
    volume::read(network)
        .map(PixelMaterial::Volume)
        .or_else(|| surface(network).map(PixelMaterial::Surface))
}

/// Whether `network` is a Pixel Volume's.
pub fn is_volume(network: &MaterialNetwork) -> bool {
    volume::read(network).is_some()
}

fn surface_type() -> NodeType {
    NodeType {
        engine: Arc::from(ID),
        name: Arc::from(SURFACE),
        label: Arc::from("Pixel Surface"),
        category: Arc::from("Surface"),
        aliases: ["pixel", "material", "surface", "flat", "toon", "sprite"]
            .map(Arc::from)
            .to_vec(),
        surface: Some(Arc::from("Pixel Surface")),
        defaults: vec![(
            Arc::from(ATTRIBUTE),
            ParamValue::Attribute(Arc::from(DEFAULT_ATTRIBUTE)),
        )],
        layout: Arc::new(layout),
    }
}

/// An opaque Pixel Surface network of `color` times the mesh attribute
/// `attribute` (empty: none).
pub fn network(color: [f32; 3], attribute: &str) -> MaterialNetwork {
    network_of(&PixelSurface {
        color,
        attribute: Arc::from(attribute),
        ..PixelSurface::default()
    })
}

/// The Pixel Surface network that renders as `surface` (its translucency
/// inputs set only where they differ from their defaults).
pub fn network_of(surface: &PixelSurface) -> MaterialNetwork {
    let opaque = PixelSurface::default();
    let mut node = ShaderNode::new(SURFACE)
        .with_param(ATTRIBUTE, ParamValue::Attribute(surface.attribute.clone()))
        .with(COLOR, Constant::Color(surface.color));
    if surface.translucency != opaque.translucency {
        node = node.with(TRANSLUCENCY, Constant::Number(surface.translucency));
    }
    if surface.translucency_color != opaque.translucency_color {
        node = node.with(
            TRANSLUCENCY_COLOR,
            Constant::Color(surface.translucency_color),
        );
    }
    MaterialNetwork::single(ID, node)
}

/// What `network` renders as: its Pixel Surface's inputs and attribute
/// (unset: the defaults, as a host fits them); none when it is not a Pixel
/// network with a surface.
pub fn surface(network: &MaterialNetwork) -> Option<PixelSurface> {
    if &*network.engine != ID {
        return None;
    }
    let node = network.nodes.iter().find(|n| &*n.kind == SURFACE)?;
    let opaque = PixelSurface::default();
    let triple = |name, default| node.value(name).map_or(default, Constant::triple);
    let attribute = match node.param(ATTRIBUTE) {
        Some(ParamValue::Attribute(name)) => name.clone(),
        _ => Arc::from(DEFAULT_ATTRIBUTE),
    };
    Some(PixelSurface {
        color: triple(COLOR, opaque.color),
        attribute,
        translucency: triple(TRANSLUCENCY, [opaque.translucency; 3])[0],
        translucency_color: triple(TRANSLUCENCY_COLOR, opaque.translucency_color),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_network_renders_its_surface_color_and_attribute() {
        assert_eq!(
            surface(&network([0.1, 0.2, 0.3], "tone")),
            Some(PixelSurface {
                color: [0.1, 0.2, 0.3],
                attribute: Arc::from("tone"),
                ..PixelSurface::default()
            })
        );
        let leaf = PixelSurface {
            color: [0.2, 0.5, 0.1],
            attribute: Arc::from("Cd"),
            translucency: 0.6,
            translucency_color: [1.2, 1.1, 0.5],
        };
        assert_eq!(surface(&network_of(&leaf)), Some(leaf.clone()));
        let t = leaf.transmittance();
        assert!((t[0] - 0.72).abs() < 1e-6 && (t[2] - 0.3).abs() < 1e-6);
        // Unset: the default color, reading Cd as a fitted node does.
        let unset = MaterialNetwork::single(ID, ShaderNode::new(SURFACE));
        let read = surface(&unset).unwrap();
        assert_eq!(read.color, PixelSurface::default().color);
        assert_eq!(&*read.attribute, DEFAULT_ATTRIBUTE);
        assert_eq!(read.transmittance(), [0.0; 3], "opaque unless asked");
        let foreign = MaterialNetwork {
            engine: Arc::from("spectre"),
            ..network([1.0; 3], "")
        };
        assert_eq!(surface(&foreign), None);
        let t = &node_types()[0];
        assert_eq!(&*node_types()[1].name, VOLUME);
        assert_eq!(t.fit(&t.defaults), t.defaults);
        let layout = t.layout(&t.defaults);
        assert_eq!(layout.input(COLOR), Some(0));
        assert_eq!(layout.input(TRANSLUCENCY), Some(1));
        assert_eq!(layout.input(TRANSLUCENCY_COLOR), Some(2));
    }

    #[test]
    fn networks_read_as_surfaces_or_volumes() {
        use crate::volume::PixelVolume;
        let fog = volume::network(&PixelVolume::default());
        assert_eq!(
            read(&fog),
            Some(PixelMaterial::Volume(PixelVolume::default()))
        );
        assert!(is_volume(&fog) && !is_volume(&network([1.0; 3], "")));
        assert!(matches!(
            read(&network([1.0; 3], "")),
            Some(PixelMaterial::Surface(_))
        ));
    }

    #[test]
    fn hosts_send_the_attribute_a_surface_reads() {
        assert_eq!(
            network([1.0; 3], "Cd").attribute_names(),
            [Arc::<str>::from("Cd")]
        );
        assert!(network([1.0; 3], "").attribute_names().is_empty());
    }
}
