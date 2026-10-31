//! Pixel renders the SDK's standard fixtures, as core's render sync hands
//! them over, exactly as its goldens (tests/conformance); a GPU test.
use kinogaki_render_api::display::{Engine, Options};
use kinogaki_studio_api::conformance::{self, Fixtures, Invariants};
use kinogaki_testkit::render::{Check, EngineConsumer};
use std::future::Future;
use std::path::Path;
use std::task::{Context, Poll, Waker};

#[test]
#[ignore = "requires a native graphics adapter; run with tools/check_all.py --gpu"]
fn pixel_renders_the_standard_fixtures_as_its_goldens() {
    let (device, queue) = device();
    let engine = kinogaki_pixel_render_wgpu::Pixel;
    let make = |wake| {
        let (engine, display) = engine.make(&device, &queue, wake, &Options::default())?;
        Ok((engine, Box::new(display) as Box<dyn std::any::Any>))
    };
    let goldens = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance");
    let check = Check::Exact;
    let consumer = EngineConsumer::new("Pixel", engine.description(), &make, check, goldens);
    let report = conformance::run(&[&Fixtures::default()], &[&consumer, &Invariants]);
    assert!(report.failures().is_empty(), "{report}");
}

/// This machine's GPU device.
fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    };
    let adapter = ready(instance.request_adapter(&options)).expect("a GPU adapter");
    let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    let descriptor = wgpu::DeviceDescriptor {
        required_features: features,
        ..Default::default()
    };
    ready(adapter.request_device(&descriptor)).expect("a device")
}

/// `future`'s output, polled to completion (wgpu's requests resolve
/// without a runtime).
fn ready<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = future.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}
