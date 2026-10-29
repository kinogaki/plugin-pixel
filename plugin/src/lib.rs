//! Pixel as a Studio plugin (`renders/pixel.dylib`): its package, the
//! engine alone.
use kinogaki_shared as _;
use kinogaki_studio_api::{Package, PackageId, export_package};

/// Pixel's package.
pub fn package() -> Package {
    Package::new(PackageId::declared("kinogaki.pixel"), "Pixel")
        .with_engine(kinogaki_pixel_render_wgpu::Pixel)
}

export_package!(package);
