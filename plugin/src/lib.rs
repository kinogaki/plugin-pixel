//! Pixel as a Studio plugin (`renders/pixel.dylib`): its package, the
//! engine and its vocabulary as Shading node types (`kinogaki.pixel.<name>`,
//! [`shaders`]), which the render package's compiler compiles, and its
//! material presets as Materials Tab makes ([`presets`]).
use kinogaki_render_api::display::Engine;
use kinogaki_shared as _;
use kinogaki_studio_api::render::{presets, shaders};
use kinogaki_studio_api::{Package, PackageId, export_package};

/// Pixel's package: the engine, its shader node types and its presets.
pub fn package() -> Package {
    let id = PackageId::declared("kinogaki.pixel");
    let engine = kinogaki_pixel_render_wgpu::Pixel;
    let mut package = Package::new(id, "Pixel");
    let description = engine.description();
    for shader in shaders(id, &description) {
        package = package.with_node_type(shader);
    }
    for preset in presets(id, &description) {
        package = package.with_creation(preset);
    }
    package.with_engine(engine)
}

export_package!(package);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_vocabulary_type_is_a_shader_named_under_the_package() {
        let vocabulary = kinogaki_pixel_render_wgpu::Pixel
            .description()
            .material_nodes;
        let p = package();
        let ids: Vec<&str> = p.node_types.iter().map(|t| t.id().as_str()).collect();
        let want: Vec<String> = (vocabulary.iter())
            .map(|t| format!("kinogaki.pixel.{}", t.name))
            .collect();
        assert!(!ids.is_empty());
        assert_eq!(ids, want);
        assert_eq!(p.engines.len(), 1);
    }

    #[test]
    fn each_preset_makes_a_material_of_a_registered_surface() {
        let p = package();
        let presets = kinogaki_pixel_render_wgpu::Pixel
            .description()
            .material_presets;
        assert_eq!(p.creations.len(), presets.len());
        let shaders: Vec<_> = p.node_types.iter().map(|t| t.id()).collect();
        for (made, preset) in p.creations.iter().zip(&presets) {
            assert_eq!(made.label, *preset.name);
            assert_eq!(
                made.node.node_type.as_str(),
                kinogaki_studio_api::render::MATERIAL
            );
            let [surface] = &made.inside[..] else {
                panic!("{} starts with its surface", made.label);
            };
            assert!(surface.display, "{}", made.label);
            assert!(shaders.contains(&surface.node_type), "{}", made.label);
        }
    }
}
