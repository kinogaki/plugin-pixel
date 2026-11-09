//! Pixel's GPU tests, each `#[ignore]`d: they need a native graphics
//! adapter (`cargo test --test gpu -- --ignored`).
mod conformance;
mod cutouts;
mod engine;
mod lights;
mod limits;
mod moves;
mod portals;
mod support;
mod textures;
