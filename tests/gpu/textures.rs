//! Pixel samples a material's texture on the GPU: an unlit square facing
//! an orthographic camera, its corners' coordinates running past the
//! picture, shows each texel where the coordinates put it (the nearest
//! texel), each axis wrapped as its node says, the placement a glTF texture
//! gives, and nothing where a cutout cuts its alpha.
use super::support::{LIMIT, at, look, pixel, unlit_cutout};
use kinogaki_pixel::sampled::sampled;
use kinogaki_render_api::geometry::image::{Encoding, Format, Image, Rect, Sample};
use kinogaki_render_api::geometry::{Attribute, Column, Domain, Meaning, Sampled, Vec3, names};
use kinogaki_render_api::value::Value;
use kinogaki_render_api::{
    Camera, Id, Material, MaterialNetwork, Prim, PrimId, Projection, SceneUpdate, Shape, Subsets,
    Topology, ViewCamera, ViewDesc, ViewId, ViewMode,
};
use std::sync::Arc;

/// The picture: two rows of two texels, red and green, then blue and a
/// magenta whose alpha is 0.
const TEXELS: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, 0.0],
];

/// The view's side, in pixels (one a film pixel).
const SIDE: u32 = 64;

fn id(slot: u32) -> PrimId {
    PrimId {
        slot,
        generation: 0,
    }
}

/// The square from (-1, -1) to (1, 1) facing +Z, its coordinates running
/// from `lo` to `hi` along each axis.
fn square(lo: f32, hi: f32) -> Shape {
    let points: Column<[f32; 3]> = vec![
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ]
    .into();
    let uv = vec![[lo, lo], [hi, lo], [hi, hi], [lo, hi]];
    let uv = Attribute::tuples(uv, Meaning::TextureCoord).unwrap();
    Shape::builder(
        points,
        Topology::Triangles(vec![[0, 1, 2], [0, 2, 3]].into()),
    )
    .primvar(Domain::Point, names::UV, uv)
    .subsets(Subsets::new(
        vec![0u16, 0].into(),
        vec![Id::<Material>::of(id(0))],
    ))
    .build()
    .unwrap()
}

/// The picture as the host sends it: premultiplied.
fn picture() -> Image {
    let format = Format::rgba(Sample::F32, Encoding::color("ACEScg", true));
    let texels: Vec<f32> = TEXELS.iter().flat_map(|t| t.map(|c| c * t[3])).collect();
    Image::builder(Rect::new([0, 0], [2, 2]).unwrap())
        .plane(format, bytemuck::cast_slice(&texels))
        .build()
        .unwrap()
}

/// The frame of `network` on a square with coordinates `lo` to `hi`, seen
/// straight on through a 4-unit-wide orthographic view: each film pixel 1/16
/// of a unit, the square from pixel 16 to 48.
fn rendered(network: MaterialNetwork, (lo, hi): (f32, f32)) -> Arc<kinogaki_render_api::Frame> {
    let (mut engine, _display, waiter, _device) = pixel();
    let material = Material {
        network: Arc::new(network),
        path: Arc::from("/Materials/Four"),
        double_sided: true,
    };
    engine.update(SceneUpdate {
        added: vec![
            (id(0), Prim::Material(material)),
            (id(1), Prim::Shape(square(lo, hi))),
            (id(2), Prim::Texture(picture())),
        ],
        ..SceneUpdate::new()
    });
    let camera = Camera {
        projection: Projection::Orthographic,
        horizontal_aperture: 4000.0,
        vertical_aperture: 4000.0,
        xform: Sampled::still(look(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO)),
        ..Camera::default()
    };
    let values = [
        ("pixel_size".to_owned(), Value::Integer(1)),
        ("dither".to_owned(), Value::Choice("off".into())),
    ];
    let view = ViewDesc {
        camera: ViewCamera::Free(camera),
        width: SIDE,
        height: SIDE,
        mode: ViewMode::Interactive,
        values: Arc::new(values.into_iter().collect()),
    };
    engine.view(ViewId(1), view);
    waiter.until(LIMIT, || engine.latest(ViewId(1)).is_some());
    engine.latest(ViewId(1)).unwrap()
}

/// The film pixel showing coordinates `(u, v)` on a square with coordinates
/// `lo` to `hi`.
fn pixel_of((u, v): (f32, f32), (lo, hi): (f32, f32)) -> (u32, u32) {
    let x = -1.0 + 2.0 * (u - lo) / (hi - lo);
    let y = -1.0 + 2.0 * (v - lo) / (hi - lo);
    ((32.0 + x * 16.0) as u32, (32.0 - y * 16.0) as u32)
}

fn near(a: [f32; 4], b: [f32; 4]) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() < 2e-3)
}

#[test]
#[ignore = "gpu"]
fn each_texel_shows_where_its_coordinates_put_it_and_a_cutout_shows_nothing() {
    let range = (0.0, 1.0);
    let frame = rendered(unlit_cutout(id(2), &[]), range);
    // v runs up, the picture's rows down: red top left.
    for (uv, texel) in [((0.25, 0.75), 0), ((0.75, 0.75), 1), ((0.25, 0.25), 2)] {
        let (x, y) = pixel_of(uv, range);
        let shown = at(&frame, x, y);
        assert!(near(shown, TEXELS[texel]), "{uv:?}: {shown:?}");
    }
    let (x, y) = pixel_of((0.75, 0.25), range);
    let cut = at(&frame, x, y);
    assert!(!near(cut, TEXELS[3]), "cut out: {cut:?}");
}

#[test]
#[ignore = "gpu"]
fn each_axis_wraps_as_its_node_says() {
    let range = (-1.0, 2.0);
    // Along u past 1, on the top row: repeat reads red again, mirror and
    // clamp green, black nothing.
    for (mode, expected) in [
        ("periodic", Some(0)),
        ("mirror", Some(1)),
        ("clamp", Some(1)),
        ("constant", None),
    ] {
        let mode_value = [("uaddressmode", Value::Choice(mode.into()))];
        let frame = rendered(unlit_cutout(id(2), &mode_value), range);
        let (x, y) = pixel_of((1.25, 0.75), range);
        let shown = at(&frame, x, y);
        match expected {
            Some(texel) => assert!(near(shown, TEXELS[texel]), "{mode}: {shown:?}"),
            None => assert!(near(shown, [0.0; 4]), "{mode}: {shown:?}"),
        }
    }
}

#[test]
#[ignore = "gpu"]
fn a_gltf_texture_rotates_as_its_graph_places_it() {
    let range = (0.0, 1.0);
    let network = unlit_cutout(id(2), &[("rotate", Value::Number(90.0))]);
    let texture = network
        .links()
        .iter()
        .find(|l| &*l.input == "emission_color")
        .unwrap()
        .from;
    let placed = sampled(&network, texture).unwrap().transform;
    let frame = rendered(network, range);
    for uv in [(0.25, 0.75), (0.75, 0.75), (0.25, 0.25)] {
        // Where the placement puts the coordinates, wrapped, is the texel.
        let st = placed.map(|[a, b, c]| a * uv.0 + b * uv.1 + c);
        let column = ((st[0].rem_euclid(1.0)) * 2.0) as usize;
        let row = (((1.0 - st[1]).rem_euclid(1.0)) * 2.0) as usize;
        let texel = row * 2 + column;
        if texel == 3 {
            continue;
        }
        let (x, y) = pixel_of(uv, range);
        let shown = at(&frame, x, y);
        assert!(near(shown, TEXELS[texel]), "{uv:?} to {st:?}: {shown:?}");
    }
}
