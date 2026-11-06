//! Pixel renders the SDK's standard values as core's render sync sends them
//! (the harness's engine consumer of the conformance suite).
use super::support::device;
use kinogaki_harness::conformance::EngineConsumer;
use kinogaki_pixel::engine::Pixel;
use kinogaki_render_api::display::{Engine, Options};
use kinogaki_studio_api_testing::{Fixtures, Invariants, run};

#[test]
#[ignore = "requires a native graphics adapter; run with tools/check_all.py --gpu"]
fn pixel_renders_every_standard_value() {
    let (device, queue) = device();
    let make = |wake| Pixel.make(&device, &queue, wake, &Options::default());
    let consumer = EngineConsumer::new("Pixel", &make);
    let report = run(&[&Fixtures::default()], &[&consumer, &Invariants]);
    eprintln!("{report}");
    assert!(report.failures().is_empty(), "{report}");
}
