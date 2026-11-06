//! What Pixel's GPU tests share: the engine on the SDK's test device with
//! its wakes counted, and a small scene as a host sends it: a grey floor, a
//! red box on it, a sun and a sky.
use kinogaki_pixel::engine::Pixel;
use kinogaki_pixel::nodes::SURFACE;
use kinogaki_render_api::display::{Display, Engine, Options, wgpu};
use kinogaki_render_api::geometry::{Column, Sampled, Vec3, Xform};
use kinogaki_render_api::standard::{Light, Shape as LightShape};
use kinogaki_render_api::value::{Parameters, Value};
use kinogaki_render_api::{
    Camera, Id, Material, MaterialNetwork, Prim, PrimId, RenderEngine, SceneUpdate, Shape, Subsets,
    Topology, ViewCamera, ViewDesc, ViewMode,
};
use kinogaki_studio_api_testing::engine::{self, Waiter};
use std::sync::Arc;
use std::time::Duration;

/// How long a test waits for Pixel.
pub const LIMIT: Duration = Duration::from_secs(60);

/// Pixel on this machine's GPU, its frames carrying their pixels.
pub fn pixel() -> (
    Box<dyn RenderEngine>,
    Box<dyn Display>,
    Waiter,
    wgpu::Device,
) {
    let (device, queue) = engine::device();
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
        double_sided: true,
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
