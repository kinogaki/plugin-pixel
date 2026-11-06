//! Pixel's GPU tests, each `#[ignore]`d: they need a native graphics
//! adapter (`cargo test --test gpu -- --ignored`).
mod conformance;
mod engine;
mod moves;
mod support;
