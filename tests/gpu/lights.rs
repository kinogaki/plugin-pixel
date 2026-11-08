//! UsdLux's kinds and APIs as Pixel lights them: a floor seen straight down
//! through an orthographic camera, one scene unit a pixel, its Direct Light
//! (off a white surface, E / π) read below a light and 45° off its axis.
//! Each shape lights as a point of its power; ShapingAPI's cone, focus and
//! IES profile shape it; a color map lights with its mean; Diffuse scales
//! it; ShadowAPI colors, switches and limits its shadows; and the camera
//! sees a visible light's shape, which shadows nothing.
use super::support::{LIMIT, id, look, pixel, surface};
use kinogaki_render_api::aov::standard as aovs;
use kinogaki_render_api::geometry::image::{
    Decoding, Encoding, Format, Image, Rect, Sample, TextureImage,
};
use kinogaki_render_api::geometry::{Column, Sampled, Vec3, Xform};
use kinogaki_render_api::standard::ies::Profile;
use kinogaki_render_api::standard::{Light, LightShape, Shadow, Shaping};
use kinogaki_render_api::value::{ImageRef, Value};
use kinogaki_render_api::{
    AovDesc, AovFormat, Camera, Id, Material, Prim, Projection, SceneUpdate, Shape, Subsets,
    Topology, ViewCamera, ViewDesc, ViewId, ViewMode,
};
use std::f32::consts::PI;
use std::sync::Arc;

/// The view's side, in pixels.
const SIDE: u32 = 24;

/// The pixel over (0.5, 0.5), below each light; the one over (4.5, 0.5), 45°
/// off the axis of a light 4 above.
const BELOW: (u32, u32) = (12, 12);
const OFF: (u32, u32) = (16, 12);

/// How high the lights hang above the floor.
const HIGH: f32 = 4.0;

/// Looking straight down from `at`, -Z the image's top.
fn down(at: Vec3) -> Xform {
    Xform::from_axes(Vec3::X, -Vec3::Z, Vec3::Y, at)
}

/// Over the pixel below, `HIGH` up, shining down.
fn over() -> Sampled<Xform> {
    Sampled::still(down(Vec3::new(0.5, f64::from(HIGH), 0.5)))
}

/// A square of half-size `h` at height `y` in material 0.
fn square(h: f32, y: f32) -> Shape {
    let points: Column<[f32; 3]> = [[-h, -h], [h, -h], [h, h], [-h, h]]
        .map(|[x, z]| [x, y, z])
        .to_vec()
        .into();
    Shape::builder(
        points,
        Topology::Triangles(vec![[0, 1, 2], [0, 2, 3]].into()),
    )
    .subsets(Subsets::new(
        vec![0u16, 0].into(),
        vec![Id::<Material>::of(id(0))],
    ))
    .build()
    .unwrap()
}

/// A picture of `channels` channels of floats, `width` × `height`.
fn picture(size: [u32; 2], channels: &[&str], encoding: Encoding, texels: &[f32]) -> TextureImage {
    let format = Format {
        name: "".into(),
        channels: channels.iter().map(|c| Arc::from(*c)).collect(),
        sample: Sample::F32,
        encoding: encoding.clone(),
    };
    let window = Rect::new([0, 0], [size[0] as i32, size[1] as i32]).unwrap();
    TextureImage {
        image: Image::builder(window)
            .plane(format, texels)
            .build()
            .unwrap(),
        decoding: (encoding != Encoding::Raw).then_some(Decoding::IDENTITY),
    }
}

/// A light as a prim, its images (by row) the texture prims at `images`.
fn prim(light: &Light, images: &[(&str, u32)]) -> Prim {
    let mut declared = light.declared();
    for (row, slot) in images {
        declared
            .textures
            .insert((*row).to_owned(), Id::of(id(*slot)));
    }
    Prim::Declared(declared)
}

/// What a view of `prims` over a floor (and a slab 1 up when `slab`) shows
/// at `pixels` in `output`, the camera `height` up.
fn shown(
    prims: Vec<Prim>,
    slab: bool,
    height: f32,
    output: &str,
    pixels: &[(u32, u32)],
) -> Vec<[f32; 4]> {
    let (mut engine, _display, waiter, _device) = pixel();
    let mut added = vec![
        (id(0), surface([0.5; 3])),
        (id(1), Prim::Shape(square(30.0, 0.0))),
    ];
    if slab {
        added.push((id(2), Prim::Shape(square(3.0, 1.0))));
    }
    added.extend((prims.into_iter().enumerate()).map(|(k, p)| (id(10 + k as u32), p)));
    engine.update(SceneUpdate {
        added,
        ..SceneUpdate::new()
    });
    let camera = Camera {
        projection: Projection::Orthographic,
        horizontal_aperture: SIDE as f32 * 1000.0,
        vertical_aperture: SIDE as f32 * 1000.0,
        xform: Sampled::still(down(Vec3::new(0.0, f64::from(height), 0.0))),
        ..Camera::default()
    };
    let values = [
        ("pixel_size", Value::Integer(1)),
        ("dither", Value::Choice("off".into())),
        ("gi", Value::Number(0.0)),
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
    let asked = [AovDesc {
        aov: engine.description().aov(output).cloned().unwrap(),
        format: AovFormat::F32,
    }];
    let read = waiter.take(&engine.finish(ViewId(1), &asked), LIMIT);
    let plane = read.plane(0).unwrap();
    let channels = plane.len() / (SIDE * SIDE * 4) as usize;
    let value = |at: usize| f32::from_ne_bytes(plane[at * 4..at * 4 + 4].try_into().unwrap());
    (pixels.iter())
        .map(|&(x, y)| {
            let first = (y * SIDE + x) as usize * channels;
            std::array::from_fn(|c| if c < channels { value(first + c) } else { 0.0 })
        })
        .collect()
}

/// The Direct Light below and 45° off lights `prims`, seen from 0.5 up
/// (the lights behind the camera).
fn direct(prims: Vec<Prim>) -> [[f32; 4]; 2] {
    let at = shown(prims, false, 0.5, aovs::DIRECT, &[BELOW, OFF]);
    [at[0], at[1]]
}

fn near(shown: f32, expected: f32) -> bool {
    (shown - expected).abs() <= 1e-3 * expected.abs().max(1e-3)
}

#[test]
#[ignore = "gpu"]
fn each_shape_lights_the_floor_below_as_a_point_of_its_power() {
    let d2 = HIGH * HIGH;
    let light = |shape: LightShape, normalize: bool| Light {
        shape,
        intensity: 2.0,
        normalize,
        xform: over(),
        ..Light::default()
    };
    let sphere = |point| LightShape::Sphere { radius: 0.5, point };
    let tube = |line| LightShape::Cylinder {
        length: 2.0,
        radius: 0.25,
        line,
    };
    let rect = LightShape::Rect {
        width: 1.0,
        height: 1.0,
        texture: None,
    };
    let disk = LightShape::Disk { radius: 0.5 };
    // E / π: a sphere's intensity L π r², a tube's across its axis L 2r
    // length, an area light's power L A falling off as 1 / (d² + A / π).
    let cases = [
        ("sphere", light(sphere(false), false), 2.0 * 0.25 / d2),
        ("point", light(sphere(true), false), 2.0 * 0.25 / d2),
        (
            "normalized sphere",
            light(sphere(false), true),
            0.5 / d2 / PI,
        ),
        (
            "cylinder",
            light(tube(false), false),
            2.0 * 0.5 * 2.0 / d2 / PI,
        ),
        ("line", light(tube(true), false), 2.0 * 0.5 * 2.0 / d2 / PI),
        ("rect", light(rect, false), 2.0 / (d2 + 1.0 / PI) / PI),
        (
            "disk",
            light(disk, false),
            2.0 * PI * 0.25 / (d2 + 0.25) / PI,
        ),
    ];
    for (what, light, expected) in cases {
        let [below, _] = direct(vec![prim(&light, &[])]);
        println!("{what}: {below:?}, expected {expected}");
        assert!(near(below[1], expected), "{what}: {below:?} for {expected}");
    }
}

#[test]
#[ignore = "gpu"]
fn shaping_cones_focus_and_profiles_shape_the_light() {
    let disk = |shaping: Option<Shaping>| Light {
        shape: LightShape::Disk { radius: 0.25 },
        intensity: 20.0,
        shaping,
        xform: over(),
        ..Light::default()
    };
    let [plain, plain_off] = direct(vec![prim(&disk(None), &[])]);
    let cone = Shaping {
        cone_angle: 20.0,
        ..Shaping::default()
    };
    let [below, off] = direct(vec![prim(&disk(Some(cone)), &[])]);
    println!("cone: below {below:?} (unshaped {plain:?}), 45° off {off:?}");
    assert!(near(below[1], plain[1]) && off[1] == 0.0);
    // Focus 4 toward blue: 45° off, red and green keep cos⁴ = ¼.
    let focus = Shaping {
        cone_angle: 180.0,
        focus: 4.0,
        focus_tint: [0.0, 0.0, 1.0],
        ..Shaping::default()
    };
    let [_, off] = direct(vec![prim(&disk(Some(focus)), &[])]);
    println!("focus: 45° off {off:?} (unshaped {plain_off:?})");
    assert!(near(off[0], 0.25 * plain_off[0]) && near(off[2], plain_off[2]));
    // An IES profile of 1000 cd within 30° of its nadir, none past 40°.
    let text = "IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 4 1 1 2 0 0 0\n1 1 0\n\
        0 30 40 90\n0\n1000 1000 0 0\n";
    let grid = Profile::parse(text).unwrap().grid();
    let profile = picture(
        [grid.width, grid.height],
        &["Y", "A"],
        Encoding::Raw,
        &grid.texels,
    );
    let ies = Shaping {
        cone_angle: 180.0,
        ies: Some(ImageRef::File(
            kinogaki_render_api::geometry::image::ImageSource::path("lamp.ies"),
        )),
        ..Shaping::default()
    };
    let point = Light {
        shape: LightShape::Sphere {
            radius: 0.1,
            point: true,
        },
        intensity: 1.0,
        shaping: Some(ies),
        xform: over(),
        ..Light::default()
    };
    let [below, off] = direct(vec![
        Prim::Texture(profile),
        prim(&point, &[("iesFile", 10)]),
    ]);
    // 1000 cd × the point's π r² over d², over π; 45° off, past 40°, what
    // the profile's 16 angles blur across its edge.
    let expected = 1000.0 * PI * 0.01 / (HIGH * HIGH) / PI;
    println!("IES: below {below:?}, expected {expected}; 45° off {off:?}");
    assert!(near(below[1], expected) && off[1] < 0.05 * below[1]);
}

#[test]
#[ignore = "gpu"]
fn a_color_map_lights_with_its_mean_and_diffuse_scales_the_light() {
    let rect = |texture: bool, diffuse: f32| Light {
        shape: LightShape::Rect {
            width: 1.0,
            height: 1.0,
            texture: texture.then(|| {
                ImageRef::File(kinogaki_render_api::geometry::image::ImageSource::path(
                    "map.png",
                ))
            }),
        },
        intensity: 4.0,
        diffuse,
        xform: over(),
        ..Light::default()
    };
    let [white, _] = direct(vec![prim(&rect(false, 1.0), &[])]);
    // A red texel and a blue one: a mean of half red, half blue.
    let map = picture(
        [2, 1],
        &["R", "G", "B", "A"],
        Encoding::color("ACEScg", false),
        &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    );
    let [mapped, _] = direct(vec![
        Prim::Texture(map),
        prim(&rect(true, 1.0), &[("texture", 10)]),
    ]);
    println!("color map: {mapped:?} against white {white:?}");
    assert!(near(mapped[0], 0.5 * white[0]) && mapped[1] == 0.0 && near(mapped[2], 0.5 * white[2]));
    let [half, _] = direct(vec![prim(&rect(false, 0.5), &[])]);
    assert!(near(half[1], 0.5 * white[1]), "Diffuse 0.5: {half:?}");
}

#[test]
#[ignore = "gpu"]
fn shadow_api_colors_switches_and_limits_shadows() {
    // A sun of illuminance 2 straight down over a slab 1 above the floor.
    let under = |shadow: Shadow| {
        let sun = Light {
            shape: LightShape::Distant { angle: 0.0 },
            intensity: 2.0,
            shadow,
            xform: over(),
            ..Light::default()
        };
        shown(vec![prim(&sun, &[])], true, 0.5, aovs::DIRECT, &[BELOW])[0]
    };
    let lit = 2.0 / PI;
    let black = under(Shadow::default());
    let red = under(Shadow {
        color: [1.0, 0.0, 0.0],
        ..Shadow::default()
    });
    let off = under(Shadow {
        enable: false,
        ..Shadow::default()
    });
    let near_only = under(Shadow {
        distance: 0.5,
        ..Shadow::default()
    });
    // The slab 1 away, halfway into a falloff of 2 before a distance of 2.
    let faded = under(Shadow {
        distance: 2.0,
        falloff: 2.0,
        ..Shadow::default()
    });
    println!("black {black:?}, red {red:?}, off {off:?}, near {near_only:?}, faded {faded:?}");
    assert!(black[1] == 0.0);
    assert!(near(red[0], lit) && red[1] == 0.0);
    assert!(near(off[1], lit) && near(near_only[1], lit));
    // Measured from where the shadow ray starts, a step off the floor
    // toward the sun: the slab a little nearer than 1.
    assert!((faded[1] / (0.5 * lit) - 1.0).abs() < 0.03);
}

#[test]
#[ignore = "gpu"]
fn the_camera_sees_a_visible_lights_shape_which_shadows_nothing() {
    // From 10 up: a sphere 5 up over the pixel below; a sun travelling
    // down and along +X, whose shadow ray from (5.5, 0, 0.5) crosses it.
    let side = (17, 12);
    let scene = |camera: bool| {
        let sphere = Light {
            shape: LightShape::Sphere {
                radius: 1.0,
                point: false,
            },
            intensity: 0.75,
            camera,
            xform: Sampled::still(down(Vec3::new(0.5, 5.0, 0.5))),
            ..Light::default()
        };
        let sun = Light {
            shape: LightShape::Distant { angle: 0.0 },
            intensity: 2.0,
            xform: Sampled::still(look(Vec3::new(0.0, 5.0, 0.0), Vec3::new(1.0, 4.0, 0.0))),
            ..Light::default()
        };
        vec![prim(&sphere, &[]), prim(&sun, &[])]
    };
    let beauty = |camera| shown(scene(camera), false, 10.0, aovs::BEAUTY, &[BELOW])[0];
    let lit = |camera| shown(scene(camera), false, 10.0, aovs::DIRECT, &[side])[0];
    let (seen, unseen) = (beauty(true), beauty(false));
    println!("the sphere seen {seen:?}, unseen {unseen:?}");
    assert!(seen[..3].iter().all(|&v| near(v, 0.75)));
    assert!(!near(unseen[1], 0.75));
    let (seen, unseen) = (lit(true), lit(false));
    println!("beside it, its glow seen {seen:?}, unseen {unseen:?}");
    assert!(near(seen[1], unseen[1]) && unseen[1] > 2.0 / PI * 0.5);
}
