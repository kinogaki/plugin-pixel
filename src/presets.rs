//! Pixel's starting materials: a Material with its inside seeded, as Tab and
//! the menus list them under Pixel.
use crate::nodes::{SURFACE, VOLUME};
use kinogaki_render_api::value::Value;
use kinogaki_studio_api::menu::Home;
use kinogaki_studio_api::network::{NewNode, NodeCreation};
use kinogaki_studio_api::render::material_creation;
use kinogaki_studio_api::{CreationId, NodeTypeId};

/// A preset: its id, label, words search takes, node and values.
struct Preset {
    id: CreationId,
    label: &'static str,
    aliases: &'static [&'static str],
    node: NodeTypeId,
    values: &'static [(&'static str, Value)],
}

macro_rules! preset {
    ($id:literal, $label:literal, $node:expr, [$($alias:literal),*], $values:expr) => {
        Preset {
            id: CreationId::declared(concat!("kinogaki.pixel.", $id)),
            label: $label,
            aliases: &[$($alias),*],
            node: $node,
            values: $values,
        }
    };
}

const PRESETS: [Preset; 5] = [
    preset!(
        "grass",
        "Grass",
        SURFACE,
        ["green", "leaf", "foliage"],
        &[("color", Value::Color([0.12, 0.32, 0.06]))]
    ),
    preset!(
        "stone",
        "Stone",
        SURFACE,
        ["rock", "grey", "gray"],
        &[("color", Value::Color([0.32, 0.31, 0.3]))]
    ),
    preset!(
        "clay",
        "Clay",
        SURFACE,
        ["terracotta", "brick", "orange"],
        &[("color", Value::Color([0.55, 0.22, 0.1]))]
    ),
    // Even fog filling a box or any closed shape, its edges sharp (the
    // scene's Fog object is the global kind).
    preset!(
        "fog_volume",
        "Fog Volume",
        VOLUME,
        ["fog material", "bounded fog", "haze", "volume", "smoke"],
        &[
            ("density", Value::Number(0.3)),
            ("scatter_color", Value::Color([0.8, 0.82, 0.88])),
            ("anisotropy", Value::Number(0.3)),
        ]
    ),
    // Low mist for a box over the ground: forward scattering for light
    // shafts, faded in from its faces.
    preset!(
        "mist",
        "Mist",
        VOLUME,
        ["haze", "god rays", "volume"],
        &[
            ("density", Value::Number(0.15)),
            ("scatter_color", Value::Color([0.85, 0.88, 0.95])),
            ("anisotropy", Value::Number(0.5)),
            ("edge_falloff", Value::Number(0.5)),
        ]
    ),
];

/// The creations, in the order Tab lists them.
pub fn creations() -> Vec<NodeCreation> {
    (PRESETS.iter())
        .map(|preset| {
            let mut inside = NewNode::new(preset.node);
            for (name, value) in preset.values {
                inside = inside.with(*name, value.clone());
            }
            material_creation(
                preset.id,
                preset.label,
                Home::family(),
                preset.aliases,
                inside,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_preset_makes_a_material_of_a_pixel_node() {
        let made = creations();
        assert_eq!(made.len(), PRESETS.len());
        for creation in &made {
            let [inside] = &creation.inside.nodes[..] else {
                panic!("{} starts with its node", creation.label);
            };
            assert!(
                [SURFACE, VOLUME].contains(&inside.node_type),
                "{}",
                creation.label
            );
        }
    }
}
