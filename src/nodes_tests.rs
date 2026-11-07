//! Pixel's material nodes, read and translated (`nodes.rs`).
use super::*;

/// OpenPBR Surface's signature.
const OPEN_PBR: &str = "ND_open_pbr_surface_surfaceshader";

fn of(def: NodeDef, values: &[(&str, Value)]) -> MaterialNetwork {
    let values: Parameters = values
        .iter()
        .map(|(n, v)| ((*n).to_owned(), v.clone()))
        .collect();
    let mut built = MaterialNetwork::builder();
    let at = built.node(&Arc::new(def), &values).unwrap();
    built.build(at).unwrap()
}

#[test]
fn a_surface_reads_its_constants_and_its_attribute() {
    let leaf = of(
        surface(),
        &[
            ("color", Value::Color([0.2, 0.5, 0.1])),
            ("translucency", Value::Number(0.5)),
            ("translucency_color", Value::Color([1.0, 0.5, 0.0])),
        ],
    );
    let Material::Surface(s, look) = material(&leaf) else {
        panic!("a surface");
    };
    assert_eq!(
        (s.color, s.transmittance()),
        ([0.2, 0.5, 0.1], [0.5, 0.25, 0.0])
    );
    assert_eq!(
        look.attribute.as_deref(),
        Some("Cd"),
        "the default attribute"
    );
    assert!(look.texture.is_none());
    assert_eq!(
        leaf.primvars().iter().map(|p| &**p).collect::<Vec<_>>(),
        ["Cd"]
    );
    let shown = Rules.display(&leaf);
    assert_eq!(
        (shown.color, shown.primvar.as_deref(), shown.volume),
        ([0.2, 0.5, 0.1], Some("Cd"), false)
    );
    assert_eq!(material(&MaterialNetwork::default()), Material::default());
}

#[test]
fn a_volume_reads_its_medium_and_shows_as_one() {
    let mist = of(
        volume(),
        &[
            ("density", Value::Number(0.15)),
            ("emission", Value::Number(2.0)),
            ("emission_color", Value::Color([1.0, 0.5, 0.0])),
            ("noise_octaves", Value::Integer(5)),
        ],
    );
    let Material::Volume(m) = material(&mist) else {
        panic!("a volume");
    };
    assert_eq!(
        (m.density, m.emission, m.noise_octaves),
        (0.15, [2.0, 1.0, 0.0], 5)
    );
    let shown = Rules.display(&mist);
    assert!(shown.volume && shown.color == [0.8; 3]);
}

/// `values` by name.
fn values(values: &[(&str, Value)]) -> Parameters {
    (values.iter())
        .map(|(n, v)| ((*n).to_owned(), v.clone()))
        .collect()
}

/// A node feeding a surface: its signature, its values and the input
/// it feeds.
type Fed<'a> = (&'a str, &'a [(&'a str, Value)], &'a str);

/// MaterialX node `signature` holding `held`, shown by a Surface
/// Material, `fed` wired into it.
fn shown(signature: &str, held: &[(&str, Value)], fed: Option<Fed<'_>>) -> MaterialNetwork {
    let mut built = MaterialNetwork::builder();
    let s = (built.node(&mtlx::def(signature).unwrap(), &values(held))).unwrap();
    if let Some((by, held, into)) = fed {
        let n = built.node(&mtlx::def(by).unwrap(), &values(held)).unwrap();
        built.link(n, "out", s, into).unwrap();
    }
    let m = built
        .node(
            &mtlx::def("ND_surfacematerial").unwrap(),
            &Parameters::new(),
        )
        .unwrap();
    built.link(s, "out", m, "surfaceshader").unwrap();
    built.build(m).unwrap()
}

#[test]
fn surface_unlit_is_an_unlit_surface_of_its_emission() {
    let lamp = shown(
        "ND_surface_unlit",
        &[
            ("emission", Value::Number(2.0)),
            ("emission_color", Value::Color([0.5, 0.25, 1.0])),
            ("transmission", Value::Number(0.5)),
        ],
        None,
    );
    let Material::Surface(s, look) = material(&lamp) else {
        panic!("a surface");
    };
    assert!(s.unlit && look == Look::default());
    assert_eq!((s.color, s.transmittance()), ([1.0, 0.5, 2.0], [0.5; 3]));
}

#[test]
fn the_standard_form_of_light_alone_is_unlit_and_an_attribute_drives_the_color() {
    let glow = [
        ("base_weight", Value::Number(0.0)),
        ("specular_weight", Value::Number(0.0)),
        ("emission_luminance", Value::Number(1.0)),
        ("emission_color", Value::Color([0.9, 0.3, 0.1])),
    ];
    let glow = Rules.translate(&shown(OPEN_PBR, &glow, None)).unwrap();
    let Material::Surface(s, _) = material(&glow) else {
        panic!("a surface");
    };
    assert!(s.unlit);
    assert_eq!(s.color, [0.9, 0.3, 0.1]);
    let paint = [("geomprop", Value::Text("paint".to_owned()))];
    let paint = Some(("ND_geompropvalue_color3", &paint[..], "base_color"));
    let painted = shown(OPEN_PBR, &[], paint);
    let Material::Surface(s, look) = material(&Rules.translate(&painted).unwrap()) else {
        panic!("a surface");
    };
    assert_eq!(
        (s.color, look.attribute.as_deref()),
        ([1.0; 3], Some("paint"))
    );
}

#[test]
fn materialx_reaches_pixel_as_surface_unlit_or_its_standard_forms_surface() {
    use kinogaki_render_api::Engines;
    let pixel = crate::description::description();
    let engines = Engines::new(vec![pixel.clone()]);
    // glTF PBR: a Pixel Surface of its standard form's base color.
    let orange = [("base_color", Value::Color([0.9, 0.5, 0.1]))];
    let gltf = shown("ND_gltf_pbr_surfaceshader", &orange, None);

    let Material::Surface(s, _) = material(&engines.adopt(&gltf, pixel.id())) else {
        panic!("a surface");
    };
    assert_eq!((s.color, s.unlit), ([0.9, 0.5, 0.1], false));
    // USD Preview Surface: lit, its diffuse color, its emission and
    // cutout aside (a Pixel Surface has neither).
    let leaf = [
        ("diffuseColor", Value::Color([0.2, 0.6, 0.3])),
        ("emissiveColor", Value::Color([0.5, 0.5, 0.5])),
        ("opacityThreshold", Value::Number(0.5)),
    ];
    let usd = shown("ND_UsdPreviewSurface_surfaceshader", &leaf, None);
    let Material::Surface(s, _) = material(&engines.adopt(&usd, pixel.id())) else {
        panic!("a surface");
    };
    assert_eq!((s.color, s.unlit), ([0.2, 0.6, 0.3], false));
    // Surface Unlit of an image: as it is, the image sampled.
    let image = Some(("ND_image_color3", &[][..], "emission_color"));
    let lamp = engines.adopt(&shown("ND_surface_unlit", &[], image), pixel.id());
    let Material::Surface(s, look) = material(&lamp) else {
        panic!("a surface");
    };
    assert!(s.unlit && look.texture.is_some() && lamp.cut().is_empty());
}

#[test]
fn a_surface_is_an_openpbr_surface_of_its_color_and_a_volume_none() {
    let color = Value::Color([0.25, 0.5, 0.125]);
    let mine = of(surface(), &[("color", color.clone())]);
    let standard = Rules.standard(&mine).unwrap();
    let shown = &standard.nodes()[standard.surface().unwrap() as usize];
    assert_eq!(shown.def().mtlx().unwrap().node, "open_pbr_surface");
    assert_eq!(shown.values()["base_color"], color);
    assert!(Rules.standard(&of(volume(), &[])).is_none());
}

#[test]
fn the_standard_form_becomes_a_surface_of_its_base_color() {
    let color = [("base_color".to_owned(), Value::Color([0.1, 0.2, 0.3]))];
    let standard = kinogaki_render_api::material::standard::open_pbr(&color.into()).unwrap();
    let mine = Rules.translate(&standard).unwrap();
    let Material::Surface(s, _) = material(&mine) else {
        panic!("a surface");
    };
    assert_eq!(s.color, [0.1, 0.2, 0.3]);
}

/// glTF PBR whose base color is a texture rotated a quarter turn and
/// clamped, its alpha cut out at `cutoff` (glTF's mask) when given.
fn gltf_textured(cutoff: Option<f64>) -> MaterialNetwork {
    use kinogaki_render_api::geometry::image::ImageSource;
    let mut built = MaterialNetwork::builder();
    let mut mask = vec![];
    if let Some(cutoff) = cutoff {
        mask = vec![
            ("alpha_mode", Value::Integer(1)),
            ("alpha_cutoff", Value::Number(cutoff)),
        ];
    }
    let def = mtlx::def("ND_gltf_pbr_surfaceshader").unwrap();
    let s = built.node(&def, &values(&mask)).unwrap();
    let texture = [
        ("rotate", Value::Number(90.0)),
        ("uaddressmode", Value::Choice("clamp".into())),
        ("filtertype", Value::Choice("closest".into())),
        ("color", Value::Numbers(vec![0.5, 0.5, 0.5, 1.0])),
    ];
    let def = mtlx::def("ND_kinogaki_gltf_colorimage").unwrap();
    let t = built.node(&def, &values(&texture)).unwrap();
    built
        .image(t, "file", ImageSource::path("/t/leaf.png"))
        .unwrap();
    built.link(t, "outcolor", s, "base_color").unwrap();
    built.link(t, "outa", s, "alpha").unwrap();
    let m = (built.node(
        &mtlx::def("ND_surfacematerial").unwrap(),
        &Parameters::new(),
    ))
    .unwrap();
    built.link(s, "out", m, "surfaceshader").unwrap();
    built.build(m).unwrap()
}

#[test]
fn a_gltf_texture_reaches_pixel_as_its_surfaces_texture_with_its_cutout() {
    use kinogaki_render_api::Engines;
    let pixel = crate::description::description();
    let engines = Engines::new(vec![pixel.clone()]);
    for cutoff in [None, Some(0.4)] {
        let adopted = engines.adopt(&gltf_textured(cutoff), pixel.id());
        assert!(adopted.cut().is_empty(), "{:?}", adopted.cut());
        let Material::Surface(s, look) = material(&adopted) else {
            panic!("a surface");
        };
        assert_eq!((s.color, s.unlit), ([1.0; 3], false));
        let t = look.texture.expect("its base color's texture");
        assert_eq!(t.source.as_ref().unwrap().name(), "/t/leaf.png");
        assert_eq!((t.wrap, t.nearest), ([Wrap::Clamp, Wrap::Repeat], true));
        assert_eq!(t.factor, [0.5, 0.5, 0.5, 1.0]);
        assert_ne!(t.transform, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "rotated");
        assert_eq!(look.cutout, cutoff.map(|c| c as f32));
        // The viewport shows its texture through the same placement.
        let shown = Rules.display(&adopted);
        let texture = shown.texture.expect("shown");
        assert_eq!(texture.transform, t.transform);
    }
}

#[test]
fn surface_unlit_cutout_cuts_its_texture_out_in_its_mask_mode() {
    use kinogaki_render_api::geometry::image::ImageSource;
    let mut built = MaterialNetwork::builder();
    let held = [
        ("alpha_mode", Value::Integer(1)),
        ("alpha_cutoff", Value::Number(0.25)),
    ];
    let s = (built.node(
        &mtlx::def("ND_kinogaki_surface_unlit").unwrap(),
        &values(&held),
    ))
    .unwrap();
    let t = (built.node(
        &mtlx::def("ND_kinogaki_gltf_colorimage").unwrap(),
        &Parameters::new(),
    ))
    .unwrap();
    built
        .image(t, "file", ImageSource::path("/t/sprite.png"))
        .unwrap();
    built.link(t, "outcolor", s, "emission_color").unwrap();
    built.link(t, "outa", s, "opacity").unwrap();
    let sprite = built.build(s).unwrap();
    let Material::Surface(s, look) = material(&sprite) else {
        panic!("a surface");
    };
    assert!(s.unlit && look.texture.is_some());
    assert_eq!(look.cutout, Some(0.25));
}

#[test]
fn a_textured_surface_is_an_openpbr_surface_of_its_texture() {
    use kinogaki_render_api::geometry::image::ImageSource;
    let mut built = MaterialNetwork::builder();
    let s = built
        .node(
            &Arc::new(surface()),
            &values(&[("cutout", Value::Number(0.5))]),
        )
        .unwrap();
    let t = (built.node(&mtlx::def("ND_image_color3").unwrap(), &Parameters::new())).unwrap();
    built
        .image(t, "file", ImageSource::path("/t/brick.png"))
        .unwrap();
    built.link(t, "out", s, "color").unwrap();
    let mine = built.build(s).unwrap();
    let Material::Surface(_, look) = material(&mine) else {
        panic!("a surface");
    };
    assert_eq!(look.cutout, Some(0.5));
    let standard = Rules.standard(&mine).unwrap();
    let surface = standard.surface().unwrap();
    let fed = (standard.links().iter()).find(|l| l.to == surface && &*l.input == "base_color");
    let from = &standard.nodes()[fed.expect("its texture").from as usize];
    assert_eq!(from.def().mtlx().unwrap().node, "image");
    assert_eq!(from.images()["file"].name(), "/t/brick.png");
}
