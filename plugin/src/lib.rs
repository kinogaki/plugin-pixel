//! Pixel as a Studio plugin library (`renders/pixel.dylib`).
use kinogaki_studio_api::export_package;

kinogaki_shared::link!();
export_package!(kinogaki_pixel::package);
