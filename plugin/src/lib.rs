//! Pixel as a Studio plugin (`renders/pixel.dylib`): its package, the
//! engine and its vocabulary as Shading node types (`kinogaki.pixel.<name>`,
//! [`shaders`]), which the render package's compiler compiles.
use kinogaki_render_api::display::Engine;
use kinogaki_shared as _;
use kinogaki_studio_api::render::shaders;
use kinogaki_studio_api::{Package, PackageId, export_package};

/// Pixel's package: the engine and its shader node types.
pub fn package() -> Package {
    let id = PackageId::declared("kinogaki.pixel");
    let engine = kinogaki_pixel_render_wgpu::Pixel;
    let mut package = Package::new(id, "Pixel");
    for shader in shaders(id, &engine.description()) {
        package = package.with_node_type(shader);
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
}
