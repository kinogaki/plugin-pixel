//! Pixel as a host sees it before making one: what it renders, its
//! settings, its material nodes and its outputs.
use crate::outputs::Output;
use crate::{lights, nodes, settings};
use kinogaki_render_api::{AovSpec, Capabilities, EngineDescription, EngineId, PrimKind, standard};
use std::sync::Arc;

/// The engine's id.
pub const ENGINE: EngineId = EngineId::declared("kinogaki.pixel.engine");

/// Pixel, described.
pub fn description() -> EngineDescription {
    let capabilities = Capabilities {
        prims: vec![
            PrimKind::Triangles,
            PrimKind::Instancer,
            PrimKind::Camera,
            PrimKind::Material,
        ],
        // Its lights (the standard ones, gobos on distant lights and spots,
        // the rim light) and the standard fog.
        types: [lights::light_types(), vec![standard::fog()]].concat(),
        // Every output, each a film-size plane, asked for by default.
        aovs: Output::ALL.map(|o| AovSpec::new(o.aov(), true)).to_vec(),
        // Each pass is a whole frame; a pixel-art frame is drawn sharp, with
        // no motion blur.
        realtime: true,
        ..Capabilities::default()
    };
    EngineDescription::builder(ENGINE, "Pixel")
        .capabilities(capabilities)
        .settings(settings::groups(), settings::descriptors())
        .materials(nodes::defs(), Arc::new(nodes::Rules))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_declares_its_lights_the_fog_and_every_output() {
        let d = description();
        assert_eq!((&*d.label().clone(), d.id()), ("Pixel", ENGINE));
        let ids: Vec<&str> = d.capabilities().types.iter().map(|t| &*t.id).collect();
        assert_eq!(ids.last(), Some(&standard::FOG));
        assert!(ids.contains(&lights::RIM_LIGHT));
        assert!(d.capabilities().aovs[0].aov.is_beauty());
        assert!(d.aov("ambient").is_some() && d.aov("objectId").is_some());
        assert!(d.node_def(nodes::SURFACE).is_some() && d.node_def(nodes::VOLUME).is_some());
        assert!(d.capabilities().realtime && !d.capabilities().motion.any());
    }
}
