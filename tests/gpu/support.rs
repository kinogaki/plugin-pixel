//! What Pixel's GPU tests share: the device (taking the test's turn on the
//! GPU), the engine made on it with its wakes counted, and a small scene as
//! a host sends it: a grey floor, a red box on it, a sun and a sky.
use kinogaki_pixel::engine::Pixel;
use kinogaki_pixel::nodes::SURFACE;
use kinogaki_render_api::display::{Display, Engine, Options, wgpu};
use kinogaki_render_api::geometry::{Column, Sampled, Vec3, Xform};
use kinogaki_render_api::standard::{Light, Shape as LightShape};
use kinogaki_render_api::value::{Parameters, Value};
use kinogaki_render_api::{
    Camera, Id, Material, MaterialNetwork, Pending, Prim, PrimId, RenderEngine, SceneUpdate, Shape,
    Subsets, Topology, ViewCamera, ViewDesc, ViewMode, Wake,
};
use std::future::Future;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

/// `future`'s output, polled to completion (wgpu's requests resolve without
/// a runtime).
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

/// This machine's GPU, its turn taken for the test's thread.
pub fn device() -> (wgpu::Device, wgpu::Queue) {
    kinogaki_gpu_turns::share();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    };
    let adapter = ready(instance.request_adapter(&options)).expect("a GPU adapter");
    ready(adapter.request_device(&wgpu::DeviceDescriptor::default())).expect("a device")
}

/// Wakes counted, and a way to wait for them.
#[derive(Clone, Default)]
pub struct Waiter {
    state: Arc<(Mutex<u64>, Condvar)>,
}

impl Waiter {
    /// The engine's wake.
    pub fn wake(&self) -> Wake {
        let state = self.state.clone();
        Arc::new(move || {
            *state.0.lock().unwrap() += 1;
            state.1.notify_all();
        })
    }

    /// Wakes so far.
    pub fn count(&self) -> u64 {
        *self.state.0.lock().unwrap()
    }

    /// Waits (a minute at most) until `done` holds, asking after each wake.
    pub fn until(&self, mut done: impl FnMut() -> bool) {
        let start = Instant::now();
        let mut guard = self.state.0.lock().unwrap();
        while !done() {
            let left = Duration::from_secs(60).saturating_sub(start.elapsed());
            assert!(!left.is_zero(), "timed out waiting for Pixel");
            guard = self
                .state
                .1
                .wait_timeout(guard, left.min(Duration::from_millis(50)))
                .unwrap()
                .0;
        }
    }

    /// `pending`'s outcome, once resolved.
    pub fn take<T>(&self, pending: &Pending<T>) -> T {
        let mut out = None;
        self.until(|| {
            out = pending.take();
            out.is_some()
        });
        out.unwrap().expect("an outcome")
    }
}

/// Pixel on this machine's GPU, its frames carrying their pixels.
pub fn pixel() -> (
    Box<dyn RenderEngine>,
    Box<dyn Display>,
    Waiter,
    wgpu::Device,
) {
    let (device, queue) = device();
    let waiter = Waiter::default();
    let options = Options {
        read_pixels: true,
        ..Options::default()
    };
    let (engine, display) = Pixel
        .make(&device, &queue, waiter.wake(), &options)
        .expect("Pixel");
    (engine, display, waiter, device)
}

fn id(slot: u32) -> PrimId {
    PrimId {
        slot,
        generation: 0,
    }
}

/// A box of half-size `h` standing on the origin, or a floor of half-size
/// `h` when `floor`, in material `material`.
fn shape(h: f32, floor: bool, material: PrimId) -> Shape {
    let (points, triangles): (Vec<[f32; 3]>, Vec<[u32; 3]>) = if floor {
        (
            vec![[-h, 0.0, -h], [h, 0.0, -h], [h, 0.0, h], [-h, 0.0, h]],
            vec![[0, 2, 1], [0, 3, 2]],
        )
    } else {
        let p = (0..8)
            .map(|i| {
                [
                    if i & 1 == 0 { -h } else { h },
                    if i & 2 == 0 { 0.0 } else { 2.0 * h },
                    if i & 4 == 0 { -h } else { h },
                ]
            })
            .collect();
        let t = vec![
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ];
        (p, t)
    };
    let count = triangles.len();
    Shape::builder(Column::from(points), Topology::Triangles(triangles.into()))
        .subsets(Subsets::new(
            vec![0u16; count].into(),
            vec![Id::<Material>::of(material)],
        ))
        .path(if floor { "/Floor" } else { "/Box" })
        .build()
        .expect("a shape")
}

/// A Pixel Surface of `color`.
pub fn surface(color: [f64; 3]) -> Prim {
    let description = kinogaki_pixel::description::description();
    let def = description
        .node_def(SURFACE)
        .expect("the Pixel Surface")
        .clone();
    let values: Parameters = [("color".to_owned(), Value::Color(color))].into();
    let mut built = MaterialNetwork::builder();
    let at = built.node(&def, &values).expect("a node");
    Prim::Material(Material {
        network: Arc::new(built.build(at).expect("a network")),
        path: Arc::from("/Materials/Surface"),
    })
}

/// The scene's prims: materials 0 and 1, the floor 2, the box 3, a sun 4
/// and a sky 5.
pub fn scene() -> SceneUpdate {
    let sun = Light {
        shape: LightShape::Distant { angle: 0.0 },
        intensity: 3.0,
        normalize: true,
        xform: Sampled::still(look(Vec3::new(2.0, 4.0, 3.0), Vec3::ZERO)),
        ..Light::default()
    };
    let sky = Light {
        shape: LightShape::Dome {
            image: None,
            ground: Some([0.1, 0.09, 0.07]),
        },
        color: [0.32, 0.38, 0.52],
        ..Light::default()
    };
    SceneUpdate {
        added: vec![
            (id(0), surface([0.3, 0.3, 0.28])),
            (id(1), surface([0.7, 0.1, 0.05])),
            (id(2), Prim::Shape(shape(4.0, true, id(0)))),
            (id(3), Prim::Shape(shape(0.75, false, id(1)))),
            (id(4), Prim::Declared(sun.declared())),
            (id(5), Prim::Declared(sky.declared())),
        ],
        ..SceneUpdate::new()
    }
}

/// Placed at `eye`, its -Z toward `at`, +Y up.
pub fn look(eye: Vec3, at: Vec3) -> Xform {
    let back = (eye - at).normalized().expect("apart");
    let right = Vec3::Y.cross(back).normalized().expect("not straight down");
    Xform::from_axes(right, back.cross(right), back, eye)
}

/// A view of `width` × `height` through a camera at `eye` looking at the box,
/// at `values`.
pub fn view(width: u32, height: u32, eye: Vec3, values: &[(&str, Value)]) -> ViewDesc {
    let camera = Camera {
        xform: Sampled::still(look(eye, Vec3::new(0.0, 0.75, 0.0))),
        ..Camera::default()
    };
    ViewDesc {
        camera: ViewCamera::Free(camera),
        width,
        height,
        mode: ViewMode::Interactive,
        values: Arc::new(
            values
                .iter()
                .map(|(n, v)| ((*n).to_owned(), v.clone()))
                .collect(),
        ),
    }
}

/// Pixel (`x`, `y`) of a frame's half floats, as floats.
pub fn at(frame: &kinogaki_render_api::Frame, x: u32, y: u32) -> [f32; 4] {
    let i = ((y * frame.width + x) * 4) as usize;
    std::array::from_fn(|k| kinogaki_render_api::geometry::image::half_to_f32(frame.pixels[i + k]))
}
