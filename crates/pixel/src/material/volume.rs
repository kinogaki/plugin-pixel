//! The Pixel Volume node: a material's terminal that fills its mesh with a
//! participating medium ([`PixelVolume`]) instead of shading its surface.
//! Its inputs, in the inspector's order: Density, Scatter Color,
//! Anisotropy, Edge Falloff and Shadows, then Emission and Noise sections.

use std::sync::Arc;

use kinogaki_render_api::{
    Constant, InputEdit, MaterialNetwork, NodeLayout, NodeType, Params, ShaderNode, SocketKind,
    SocketSpec,
};

use super::ID;
use crate::volume::{MAX_OCTAVES, PixelVolume};

/// The Pixel Volume's type name.
pub const VOLUME: &str = "pixel_volume";

/// Its inputs' names.
pub mod input {
    /// Extinction a unit of distance.
    pub const DENSITY: &str = "density";
    /// The single-scattering albedo.
    pub const SCATTER_COLOR: &str = "scatter_color";
    /// Henyey–Greenstein g.
    pub const ANISOTROPY: &str = "anisotropy";
    /// Fade-in distance from the bounds.
    pub const EDGE_FALLOFF: &str = "edge_falloff";
    /// Dims the lights' light through it.
    pub const SHADOWS: &str = "shadows";
    /// Emitted light's strength.
    pub const EMISSION: &str = "emission";
    /// Emitted light's color.
    pub const EMISSION_COLOR: &str = "emission_color";
    /// The noise's share of the density.
    pub const NOISE_AMOUNT: &str = "noise_amount";
    /// Noise cycles a unit.
    pub const NOISE_FREQUENCY: &str = "noise_frequency";
    /// Noise octaves.
    pub const NOISE_OCTAVES: &str = "noise_octaves";
    /// Moves the noise.
    pub const NOISE_OFFSET: &str = "noise_offset";
    /// The noise's seed.
    pub const NOISE_SEED: &str = "noise_seed";
}

const EMISSION_SECTION: &str = "Emission";
const NOISE_SECTION: &str = "Noise";

/// A number socket edited over `range`, dragged over `soft`.
fn number(
    name: &'static str,
    label: &'static str,
    default: f32,
    range: (f64, f64),
    soft: (f64, f64),
    decimals: u8,
) -> SocketSpec {
    SocketSpec {
        edit: InputEdit::Number {
            range,
            soft,
            suffix: Arc::from(""),
            decimals,
        },
        ..SocketSpec::new(name, label, SocketKind::Float, [default; 3])
    }
}

fn tip(spec: SocketSpec, tooltip: &'static str) -> SocketSpec {
    SocketSpec {
        tooltip: Some(Arc::from(tooltip)),
        ..spec
    }
}

fn section(spec: SocketSpec, section: &'static str) -> SocketSpec {
    SocketSpec {
        section: Some(Arc::from(section)),
        ..spec
    }
}

/// Its layout: inputs only, no settings or outputs.
fn layout(_name: &str, _params: &Params) -> NodeLayout {
    use input::*;
    let d = PixelVolume::default();
    let big = (0.0, 1e6);
    NodeLayout {
        params: Vec::new(),
        inputs: vec![
            tip(
                number(DENSITY, "Density", d.density, big, (0.0, 4.0), 3),
                "The share of light a unit of distance takes out",
            ),
            tip(
                SocketSpec::new(
                    SCATTER_COLOR,
                    "Scatter Color",
                    SocketKind::Color,
                    d.scatter_color,
                ),
                "The share of the light taken out that scatters; the rest is absorbed",
            ),
            tip(
                number(
                    ANISOTROPY,
                    "Anisotropy",
                    d.anisotropy,
                    (-0.9, 0.9),
                    (-0.9, 0.9),
                    2,
                ),
                "Above 0 light scatters forward, brightest looking toward the light",
            ),
            tip(
                number(
                    EDGE_FALLOFF,
                    "Edge Falloff",
                    d.edge_falloff,
                    big,
                    (0.0, 10.0),
                    2,
                ),
                "How far in from the bounds the density fades in",
            ),
            tip(
                SocketSpec {
                    edit: InputEdit::Toggle,
                    ..SocketSpec::new(SHADOWS, "Shadows", SocketKind::Float, [0.0; 3])
                },
                "Dims the light passing through it, on surfaces and on itself",
            ),
            section(
                number(EMISSION, "Emission", d.emission, big, (0.0, 4.0), 3),
                EMISSION_SECTION,
            ),
            section(
                SocketSpec::new(
                    EMISSION_COLOR,
                    "Emission Color",
                    SocketKind::Color,
                    d.emission_color,
                ),
                EMISSION_SECTION,
            ),
            section(
                number(
                    NOISE_AMOUNT,
                    "Amount",
                    d.noise_amount,
                    (0.0, 1.0),
                    (0.0, 1.0),
                    3,
                ),
                NOISE_SECTION,
            ),
            section(
                number(
                    NOISE_FREQUENCY,
                    "Frequency",
                    d.noise_frequency,
                    big,
                    (0.0, 4.0),
                    3,
                ),
                NOISE_SECTION,
            ),
            section(
                number(
                    NOISE_OCTAVES,
                    "Octaves",
                    d.noise_octaves as f32,
                    (1.0, f64::from(MAX_OCTAVES)),
                    (1.0, f64::from(MAX_OCTAVES)),
                    0,
                ),
                NOISE_SECTION,
            ),
            section(
                SocketSpec::new(NOISE_OFFSET, "Offset", SocketKind::Vector, d.noise_offset),
                NOISE_SECTION,
            ),
            section(
                number(
                    NOISE_SEED,
                    "Seed",
                    d.noise_seed as f32,
                    (0.0, 9999.0),
                    (0.0, 100.0),
                    0,
                ),
                NOISE_SECTION,
            ),
        ],
        outputs: Vec::new(),
    }
}

/// The Pixel Volume's node type.
pub fn node_type() -> NodeType {
    NodeType {
        engine: Arc::from(ID),
        name: Arc::from(VOLUME),
        label: Arc::from("Pixel Volume"),
        category: Arc::from("Surface"),
        aliases: [
            "volume", "fog", "mist", "smoke", "haze", "god rays", "medium",
        ]
        .map(Arc::from)
        .to_vec(),
        surface: Some(Arc::from("Pixel Volume")),
        defaults: Vec::new(),
        layout: Arc::new(layout),
    }
}

/// The Pixel Volume network that renders as `volume` (inputs set only
/// where they differ from their defaults).
pub fn network(volume: &PixelVolume) -> MaterialNetwork {
    use input::*;
    let d = PixelVolume::default();
    let flag = |on: bool| f32::from(u8::from(on));
    let numbers = [
        (DENSITY, volume.density, d.density),
        (ANISOTROPY, volume.anisotropy, d.anisotropy),
        (EDGE_FALLOFF, volume.edge_falloff, d.edge_falloff),
        (SHADOWS, flag(volume.shadows), flag(d.shadows)),
        (EMISSION, volume.emission, d.emission),
        (NOISE_AMOUNT, volume.noise_amount, d.noise_amount),
        (NOISE_FREQUENCY, volume.noise_frequency, d.noise_frequency),
        (
            NOISE_OCTAVES,
            volume.noise_octaves as f32,
            d.noise_octaves as f32,
        ),
        (NOISE_SEED, volume.noise_seed as f32, d.noise_seed as f32),
    ]
    .map(|(name, v, default)| (name, Constant::Number(v), v != default));
    let triples = [
        (
            SCATTER_COLOR,
            Constant::Color(volume.scatter_color),
            volume.scatter_color != d.scatter_color,
        ),
        (
            EMISSION_COLOR,
            Constant::Color(volume.emission_color),
            volume.emission_color != d.emission_color,
        ),
        (
            NOISE_OFFSET,
            Constant::Vector(volume.noise_offset),
            volume.noise_offset != d.noise_offset,
        ),
    ];
    let node = numbers
        .into_iter()
        .chain(triples)
        .filter(|(_, _, set)| *set)
        .fold(ShaderNode::new(VOLUME), |node, (name, value, _)| {
            node.with(name, value)
        });
    MaterialNetwork::single(ID, node)
}

/// What `network`'s Pixel Volume renders as (unset inputs: the defaults);
/// none when it has none.
pub fn read(network: &MaterialNetwork) -> Option<PixelVolume> {
    use input::*;
    if &*network.engine != ID {
        return None;
    }
    let node = network.nodes.iter().find(|n| &*n.kind == VOLUME)?;
    let d = PixelVolume::default();
    let triple = |name, default: [f32; 3]| node.value(name).map_or(default, Constant::triple);
    let number = |name, default: f32| triple(name, [default; 3])[0];
    let count = |name, default: u32| number(name, default as f32).round().max(0.0) as u32;
    Some(
        PixelVolume {
            density: number(DENSITY, d.density),
            scatter_color: triple(SCATTER_COLOR, d.scatter_color),
            anisotropy: number(ANISOTROPY, d.anisotropy),
            emission: number(EMISSION, d.emission),
            emission_color: triple(EMISSION_COLOR, d.emission_color),
            noise_amount: number(NOISE_AMOUNT, d.noise_amount),
            noise_frequency: number(NOISE_FREQUENCY, d.noise_frequency),
            noise_octaves: count(NOISE_OCTAVES, d.noise_octaves),
            noise_offset: triple(NOISE_OFFSET, d.noise_offset),
            noise_seed: count(NOISE_SEED, d.noise_seed),
            edge_falloff: number(EDGE_FALLOFF, d.edge_falloff),
            shadows: number(SHADOWS, 0.0) > 0.5,
        }
        .clamped(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_volume_network_round_trips() {
        let mist = PixelVolume {
            density: 0.3,
            scatter_color: [0.7, 0.75, 0.9],
            anisotropy: 0.5,
            emission: 0.2,
            emission_color: [1.0, 0.5, 0.2],
            noise_amount: 0.8,
            noise_frequency: 0.25,
            noise_octaves: 4,
            noise_offset: [1.0, 2.0, 3.0],
            noise_seed: 12,
            edge_falloff: 1.5,
            shadows: true,
        };
        assert_eq!(read(&network(&mist)), Some(mist));
        let unset = MaterialNetwork::single(ID, ShaderNode::new(VOLUME));
        assert_eq!(read(&unset), Some(PixelVolume::default()));
        assert_eq!(network(&PixelVolume::default()), unset);
        assert_eq!(read(&super::super::network([1.0; 3], "")), None);
        let t = node_type();
        let layout = t.layout(&t.defaults);
        assert_eq!(layout.inputs.len(), 12);
        assert!(layout.inputs.iter().all(|s| s.shown));
    }
}
