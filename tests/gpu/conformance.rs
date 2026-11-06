//! Pixel renders what a host sends: the SDK's engine conformance.
use kinogaki_pixel::engine::Pixel;
use kinogaki_render_api::display::{Engine, Options};
use kinogaki_studio_api_testing::engine::{conformance, device};

#[test]
#[ignore = "requires a native graphics adapter; run with tools/check_all.py --gpu"]
fn pixel_renders_what_a_host_sends() {
    let (device, queue) = device();
    let make = |wake| Pixel.make(&device, &queue, wake, &Options::default());
    let report = conformance("Pixel", &make);
    eprintln!("{report}");
    assert!(report.failures().is_empty(), "{report}");
}
