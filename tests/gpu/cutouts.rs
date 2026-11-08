//! Traced rays see a cutout as the camera does: an unlit card at height 1,
//! its texture transparent where x < 0 and opaque beyond, over a floor lit
//! by a sun straight above it, seen from between them through an
//! orthographic camera looking down. Under the transparent half the floor's
//! shadow, AO, bounce and the lit fog's shadow rays meet nothing; under the
//! opaque half the card stops them all.
use super::support::{LIMIT, id, pixel, surface, unlit_cutout};
use kinogaki_render_api::aov::standard as aovs;
use kinogaki_render_api::geometry::image::{
    Decoding, Encoding, Format, Image, Rect, Sample, TextureImage,
};
use kinogaki_render_api::geometry::{
    Attribute, Column, Domain, Meaning, Sampled, Vec3, Xform, names,
};
use kinogaki_render_api::standard::{self, Light, LightShape};
use kinogaki_render_api::value::Value;
use kinogaki_render_api::{
    AovDesc, AovFormat, Camera, Declared, Id, Material, Prim, Projection, SceneUpdate, Shape,
    Subsets, Topology, ViewCamera, ViewDesc, ViewId, ViewMode,
};
use std::f32::consts::PI;
use std::sync::Arc;

/// The view's side, in pixels (one a film pixel, one scene unit a pixel).
const SIDE: u32 = 24;

/// The sun's illuminance.
const SUN: f32 = 2.0;

/// The film pixels over the floor under the card's transparent half (x =
/// -9.5) and its opaque half (x = 9.5), z = 0.5: farther from the card's
/// middle and edges than any AO or bounce ray reaches across.
const UNDER_CLEAR: (u32, u32) = (2, 12);
const UNDER_OPAQUE: (u32, u32) = (21, 12);

/// Looking straight down from `at`, -Z the image's top.
fn down(at: Vec3) -> Xform {
    Xform::from_axes(Vec3::X, -Vec3::Z, Vec3::Y, at)
}

/// A square of half-size `h` at height `y` in material `material`, its
/// texture coordinates running 0 to 1 along X and Z when `uv`.
fn square(h: f32, y: f32, material: u32, uv: bool) -> Shape {
    let corners = [[-h, -h], [h, -h], [h, h], [-h, h]];
    let points: Column<[f32; 3]> = corners.map(|[x, z]| [x, y, z]).to_vec().into();
    let mut built = Shape::builder(
        points,
        Topology::Triangles(vec![[0, 1, 2], [0, 2, 3]].into()),
    )
    .subsets(Subsets::new(
        vec![0u16, 0].into(),
        vec![Id::<Material>::of(id(material))],
    ));
    if uv {
        let st = corners.map(|[x, z]| [(x + h) / (2.0 * h), (z + h) / (2.0 * h)]);
        let st = Attribute::tuples(st.to_vec(), Meaning::TextureCoord).unwrap();
        built = built.primvar(Domain::Point, names::UV, st);
    }
    built.build().unwrap()
}

/// Two white texels, straight: the first (u < 0.5) transparent.
fn picture() -> TextureImage {
    let format = Format::rgba(Sample::F32, Encoding::color("ACEScg", false));
    let texels = [1.0f32, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let image = Image::builder(Rect::new([0, 0], [2, 1]).unwrap())
        .plane(format, &texels)
        .build()
        .unwrap();
    TextureImage {
        image,
        decoding: Some(Decoding::IDENTITY),
    }
}

/// What the floor shows at each of `pixels`, from each of `outputs`, under
/// the card, with lit fog of density 1 when `fog`.
fn rendered(fog: bool, outputs: &[&str], pixels: &[(u32, u32)]) -> Vec<Vec<[f32; 4]>> {
    let (mut engine, _display, waiter, _device) = pixel();
    let sun = Light {
        shape: LightShape::Distant { angle: 0.0 },
        intensity: SUN,
        normalize: true,
        xform: Sampled::still(down(Vec3::new(0.0, 10.0, 0.0))),
        ..Light::default()
    };
    let card = Material {
        network: Arc::new(unlit_cutout(id(2), &[])),
        path: Arc::from("/Materials/Card"),
        double_sided: true,
    };
    let mut added = vec![
        (id(0), surface([0.5; 3])),
        (id(1), Prim::Material(card)),
        (id(2), Prim::Texture(picture())),
        (id(3), Prim::Shape(square(30.0, 0.0, 0, false))),
        (id(4), Prim::Shape(square(20.0, 1.0, 1, true))),
        (id(5), Prim::Declared(sun.declared())),
    ];
    if fog {
        let mut fog = Declared::new(standard::FOG, standard::fog().defaults());
        for (name, value) in [
            ("density", Value::Number(1.0)),
            ("color", Value::Color([1.0; 3])),
            ("start", Value::Number(0.0)),
            ("height", Value::Number(0.0)),
            ("falloff", Value::Number(0.0)),
            ("inscatter", Value::Bool(true)),
            ("anisotropy", Value::Number(0.0)),
        ] {
            fog.values.insert(name.into(), value);
        }
        added.push((id(6), Prim::Declared(fog)));
    }
    engine.update(SceneUpdate {
        added,
        ..SceneUpdate::new()
    });
    let camera = Camera {
        projection: Projection::Orthographic,
        horizontal_aperture: SIDE as f32 * 1000.0,
        vertical_aperture: SIDE as f32 * 1000.0,
        xform: Sampled::still(down(Vec3::new(0.0, 0.5, 0.0))),
        ..Camera::default()
    };
    let values = [
        ("pixel_size", Value::Integer(1)),
        ("dither", Value::Choice("off".into())),
        ("ao_distance", Value::Number(10.0)),
        ("gi_distance", Value::Number(10.0)),
    ];
    let view = ViewDesc {
        camera: ViewCamera::Free(camera),
        width: SIDE,
        height: SIDE,
        mode: ViewMode::Final,
        values: Arc::new(
            (values.iter())
                .map(|(n, v)| ((*n).to_owned(), v.clone()))
                .collect(),
        ),
    };
    engine.view(ViewId(1), view);
    let asked: Vec<AovDesc> = (outputs.iter())
        .map(|o| AovDesc {
            aov: engine.description().aov(o).cloned().unwrap(),
            format: AovFormat::F32,
        })
        .collect();
    let read = waiter.take(&engine.finish(ViewId(1), &asked), LIMIT);
    (0..outputs.len())
        .map(|k| {
            let plane = read.plane(k).unwrap();
            let channels = plane.len() / (SIDE * SIDE * 4) as usize;
            let value =
                |at: usize| f32::from_ne_bytes(plane[at * 4..at * 4 + 4].try_into().unwrap());
            (pixels.iter())
                .map(|&(x, y)| {
                    let first = (y * SIDE + x) as usize * channels;
                    std::array::from_fn(|c| match c < channels {
                        true => value(first + c),
                        false => 0.0,
                    })
                })
                .collect()
        })
        .collect()
}

fn near(shown: [f32; 4], expected: f32) -> bool {
    shown[..3].iter().all(|v| (v - expected).abs() < 2e-3)
}

#[test]
#[ignore = "gpu"]
fn shadows_ao_and_bounces_pass_where_a_cutout_cuts_the_card_away() {
    let outputs = [aovs::DIRECT, aovs::OCCLUSION, aovs::INDIRECT];
    let shown = rendered(false, &outputs, &[UNDER_CLEAR, UNDER_OPAQUE]);
    let [direct, ao, bounce] = [&shown[0], &shown[1], &shown[2]];
    // The sun straight above: E / π off a white surface where it reaches.
    assert!(near(direct[0], SUN / PI), "lit: {:?}", direct[0]);
    assert!(near(direct[1], 0.0), "shadowed: {:?}", direct[1]);
    // Every AO ray gets out under the transparent half, none under the
    // opaque one.
    assert!((ao[0][0] - 1.0).abs() < 1e-3, "open: {:?}", ao[0]);
    assert!(ao[1][0].abs() < 1e-3, "closed: {:?}", ao[1]);
    // A bounce ray meeting the unlit white card brings back its light (1,
    // × GI 1); through the transparent half it meets nothing.
    assert!(near(bounce[0], 0.0), "nothing above: {:?}", bounce[0]);
    assert!(near(bounce[1], 1.0), "the card: {:?}", bounce[1]);
}

#[test]
#[ignore = "gpu"]
fn lit_fog_is_lit_where_a_cutout_cuts_the_card_away() {
    let shown = rendered(true, &[aovs::VOLUME], &[UNDER_CLEAR, UNDER_OPAQUE]);
    // Half a unit of fog of density 1 under the sun, isotropic: it sends
    // E / 4π × (1 − e^(−0.5)) toward the camera where the sun reaches it,
    // nothing under the opaque half; its opacity is the same under both.
    let opacity = 1.0 - (-0.5f32).exp();
    let lit = SUN / (4.0 * PI) * opacity;
    let [clear, opaque] = [shown[0][0], shown[0][1]];
    assert!(near(clear, lit), "lit: {clear:?} for {lit}");
    assert!(near(opaque, 0.0), "shadowed: {opaque:?}");
    for (what, alpha) in [("clear", clear[3]), ("opaque", opaque[3])] {
        assert!((alpha - opacity).abs() < 1e-3, "{what}: {alpha}");
    }
}
