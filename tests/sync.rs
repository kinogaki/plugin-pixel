//! The host's prims as Pixel's scene: shapes as meshes by material in
//! their own space, placed by instances (a move remakes the instances alone), volumes apart, tints and normals per corner,
//! lights, domes, rims and fog, and what each batch says changed.
use kinogaki_pixel::lights::{RIM_LIGHT, light_types};
use kinogaki_pixel::nodes::{SURFACE, VOLUME};
use kinogaki_pixel::sync::Sync;
use kinogaki_render_api::geometry::{
    Attribute, Column, Domain, Meaning, Sampled, Vec3, Xform, names,
};
use kinogaki_render_api::standard::{self, Light, LightShape};
use kinogaki_render_api::value::{Parameters, Value};
use kinogaki_render_api::{
    Declared, Dirty, Id, Instancer, Material, MaterialNetwork, Prim, PrimChange, PrimId,
    SceneUpdate, Shape, Subsets, Topology,
};
use std::sync::Arc;

fn id(slot: u32) -> PrimId {
    PrimId {
        slot,
        generation: 0,
    }
}

/// A unit square in the XY plane, its two triangles in `material`, with a
/// point `Cd` of `cd`.
fn square(material: Option<PrimId>, cd: [f32; 3]) -> Shape {
    let points: Column<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    .into();
    let cd = Attribute::tuples(vec![cd; 4], Meaning::Color).unwrap();
    let mut shape = Shape::builder(
        points,
        Topology::Triangles(vec![[0, 1, 2], [0, 2, 3]].into()),
    )
    .primvar(Domain::Point, names::CD, cd);
    if let Some(m) = material {
        shape = shape.subsets(Subsets::new(
            vec![0u16, 0].into(),
            vec![Id::<Material>::of(m)],
        ));
    }
    shape.build().unwrap()
}

/// A material of Pixel's node `def` holding `values`.
fn material(def: kinogaki_studio_api::NodeTypeId, values: &[(&str, Value)]) -> Prim {
    let description = kinogaki_pixel::description::description();
    let def = description.node_def(def).unwrap().clone();
    let values: Parameters = values
        .iter()
        .map(|(n, v)| ((*n).to_owned(), v.clone()))
        .collect();
    let mut built = MaterialNetwork::builder();
    let at = built.node(&def, &values).unwrap();
    Prim::Material(Material {
        network: Arc::new(built.build(at).unwrap()),
        path: Arc::from("/m"),
        double_sided: true,
    })
}

/// Instance `k`'s vertices in world space.
fn placed(scene: &kinogaki_pixel::render::Scene, k: usize) -> Vec<[f32; 3]> {
    let i = &scene.instances[k];
    let mesh = &scene.meshes[i.mesh as usize];
    let at = |p: [f32; 3]| {
        i.to_world
            .map(|r| r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3])
    };
    mesh.positions.iter().map(|&p| at(p)).collect()
}

fn added(prims: Vec<(PrimId, Prim)>) -> SceneUpdate {
    SceneUpdate {
        added: prims,
        ..SceneUpdate::new()
    }
}

#[test]
fn a_shape_is_a_world_space_mesh_tinted_by_its_material_attribute() {
    let mut sync = Sync::default();
    let red = material(SURFACE, &[("color", Value::Color([1.0, 0.0, 0.0]))]);
    let moved = Xform::translation(Vec3::new(0.0, 0.0, 5.0));
    let shape = (square(Some(id(0)), [0.5; 3]).rebuilt())
        .xform(Sampled::still(moved))
        .build()
        .unwrap();
    let changed = sync.apply(added(vec![(id(0), red), (id(1), Prim::Shape(shape))]));
    assert!(changed.meshes && changed.instances && changed.surfaces);
    let scene = sync.scene();
    let ([mesh], [instance]) = (&scene.meshes[..], &scene.instances[..]) else {
        panic!("one mesh, placed once");
    };
    // A vertex a point: its corners on a point hold the same.
    assert_eq!((mesh.positions.len(), mesh.triangles.len()), (4, 2));
    assert!(mesh.positions.iter().all(|p| p[2] == 0.0), "its own space");
    assert!(placed(scene, 0).iter().all(|p| p[2] == 5.0));
    assert_eq!(
        mesh.tints.as_deref().map(|t| t[0]),
        Some([0.5, 0.5, 0.5, 1.0]),
        "Cd, the default attribute, opaque"
    );
    assert_eq!(
        scene.surfaces[instance.surface as usize].color,
        [1.0, 0.0, 0.0]
    );
    assert_eq!(instance.id, 2, "its prim's slot, plus one");
}

/// A move remakes the instances alone: the mesh is the one made before. A
/// new point remakes it.
/// A material coming or going keeps the meshes it doesn't color, and their
/// instances take the surfaces' new indices; a hidden shape keeps its mesh
/// through being shown again.
#[test]
fn materials_and_visibility_remake_only_what_they_change() {
    let mut sync = Sync::default();
    let color = |c: [f64; 3]| material(SURFACE, &[("color", Value::Color(c))]);
    sync.apply(added(vec![
        (id(1), color([0.0, 1.0, 0.0])),
        (id(2), Prim::Shape(square(Some(id(1)), [1.0; 3]))),
    ]));
    let held = sync.scene().meshes[0].clone();
    let surface = |sync: &Sync| sync.scene().instances[0].surface as usize;
    // A material before it in index order: the green surface moves along.
    let changed = sync.apply(added(vec![(id(0), color([1.0, 0.0, 0.0]))]));
    assert!(changed.instances && !changed.meshes);
    assert!(Arc::ptr_eq(&held, &sync.scene().meshes[0]), "the mesh kept");
    let green = sync.scene().surfaces[surface(&sync)].color;
    assert_eq!(green, [0.0, 1.0, 0.0]);
    let hidden = |shown: bool| {
        let shape = square(Some(id(1)), [1.0; 3]).rebuilt();
        let visibility = kinogaki_render_api::Visibility::of(shown);
        let shape = shape.visibility(visibility).build().unwrap();
        SceneUpdate {
            changed: vec![PrimChange::new(
                id(2),
                Dirty::VISIBILITY,
                Prim::Shape(shape),
            )],
            ..SceneUpdate::new()
        }
    };
    sync.apply(hidden(false));
    assert!(sync.scene().meshes.is_empty(), "hidden");
    sync.apply(hidden(true));
    assert!(
        Arc::ptr_eq(&held, &sync.scene().meshes[0]),
        "shown with its mesh"
    );
    // The green one gone: its shape takes the default, a mesh made anew.
    let gone = SceneUpdate {
        removed: vec![id(1)],
        ..SceneUpdate::new()
    };
    sync.apply(gone);
    assert!(!Arc::ptr_eq(&held, &sync.scene().meshes[0]));
}

#[test]
fn a_moved_shape_keeps_its_mesh() {
    let mut sync = Sync::default();
    let shape = |z: f64| {
        (square(None, [1.0; 3]).rebuilt())
            .xform(Sampled::still(Xform::translation(Vec3::new(0.0, 0.0, z))))
            .build()
            .unwrap()
    };
    sync.apply(added(vec![(id(0), Prim::Shape(shape(0.0)))]));
    let before = sync.scene().meshes[0].clone();
    let moved = SceneUpdate {
        changed: vec![PrimChange::new(
            id(0),
            Dirty::TRANSFORM,
            Prim::Shape(shape(2.0)),
        )],
        ..SceneUpdate::new()
    };
    let changed = sync.apply(moved);
    assert!(changed.instances && !changed.meshes && !changed.surfaces);
    assert!(Arc::ptr_eq(&before, &sync.scene().meshes[0]));
    assert!(placed(sync.scene(), 0).iter().all(|p| p[2] == 2.0));
    let reshaped = SceneUpdate {
        changed: vec![PrimChange::new(
            id(0),
            Dirty::POINTS,
            Prim::Shape(shape(2.0)),
        )],
        ..SceneUpdate::new()
    };
    assert!(sync.apply(reshaped).meshes);
    assert!(!Arc::ptr_eq(&before, &sync.scene().meshes[0]));
}

#[test]
fn a_new_color_rewrites_the_object_table_alone() {
    let mut sync = Sync::default();
    let grey = material(SURFACE, &[]);
    sync.apply(added(vec![
        (id(0), grey),
        (id(1), Prim::Shape(square(Some(id(0)), [1.0; 3]))),
    ]));
    let blue = material(SURFACE, &[("color", Value::Color([0.0, 0.0, 1.0]))]);
    let update = SceneUpdate {
        changed: vec![PrimChange::new(id(0), Dirty::PARAMS, blue)],
        ..SceneUpdate::new()
    };
    let changed = sync.apply(update);
    assert!(changed.surfaces && !changed.meshes && !changed.instances && !changed.lights);
    assert_eq!(sync.scene().surfaces[0].color, [0.0, 0.0, 1.0]);
}

#[test]
fn a_volume_material_makes_its_shape_a_volume_never_drawn() {
    let mut sync = Sync::default();
    let fog = material(VOLUME, &[("density", Value::Number(2.0))]);
    let changed = sync.apply(added(vec![
        (id(0), fog),
        (id(1), Prim::Shape(square(Some(id(0)), [1.0; 3]))),
    ]));
    assert!(changed.volumes);
    let scene = sync.scene();
    assert!(scene.instances.is_empty());
    let [volume] = &scene.volumes[..] else {
        panic!("one volume");
    };
    assert_eq!((volume.corners.len(), volume.medium.density), (6, 2.0));
    assert_eq!(volume.own_bounds, ([0.0; 3], [1.0, 1.0, 0.0]));
    // A new density restyles it without walking the shapes.
    let denser = material(VOLUME, &[("density", Value::Number(3.0))]);
    let update = SceneUpdate {
        changed: vec![PrimChange::new(id(0), Dirty::PARAMS, denser)],
        ..SceneUpdate::new()
    };
    let changed = sync.apply(update);
    assert!(changed.volumes && !changed.meshes && !changed.instances);
    assert_eq!(sync.scene().volumes[0].medium.density, 3.0);
}

#[test]
fn a_prototype_renders_only_through_its_instancer_with_its_materials() {
    let mut sync = Sync::default();
    let blue = material(SURFACE, &[("color", Value::Color([0.0, 0.0, 1.0]))]);
    let xforms: Column<Xform> = [1.0, 3.0]
        .map(|x| Xform::translation(Vec3::new(x, 0.0, 0.0)))
        .to_vec()
        .into();
    let instancer = Instancer::builder(vec![id(1)], xforms)
        .subsets(Subsets::new(
            vec![0u16, Subsets::DEFAULT].into(),
            vec![Id::<Material>::of(id(0))],
        ))
        .build()
        .unwrap();
    sync.apply(added(vec![
        (id(0), blue),
        (id(1), Prim::Shape(square(None, [1.0; 3]))),
        (id(2), Prim::Instancer(instancer)),
    ]));
    let scene = sync.scene();
    assert_eq!(
        scene.instances.len(),
        2,
        "two instances, the prototype not on its own"
    );
    let lowest = |k: usize| {
        placed(scene, k)
            .iter()
            .map(|p| p[0])
            .fold(f32::MAX, f32::min)
    };
    let mut lows: Vec<f32> = (0..2).map(lowest).collect();
    lows.sort_by(f32::total_cmp);
    assert_eq!(lows, [1.0, 3.0]);
    let first = &scene.instances[(0..2).find(|&k| lowest(k) == 1.0).unwrap()];
    assert_eq!(
        scene.surfaces[first.surface as usize].color,
        [0.0, 0.0, 1.0],
        "the instance's material"
    );
    let second = &scene.instances[(0..2).find(|&k| lowest(k) == 3.0).unwrap()];
    assert!(
        scene.surfaces.get(second.surface as usize).is_none(),
        "the default"
    );
}

/// An instance's material map renders its prototype's red triangles in
/// blue; the other instance keeps red; both place the one shape's meshes
/// by its own materials, the mapped one through a mesh made once for its
/// map.
#[test]
fn an_instances_material_map_renders_its_prototypes_materials_as_others() {
    use kinogaki_render_api::MaterialMaps;
    let mut sync = Sync::default();
    let red = material(SURFACE, &[("color", Value::Color([1.0, 0.0, 0.0]))]);
    let blue = material(SURFACE, &[("color", Value::Color([0.0, 0.0, 1.0]))]);
    let xforms: Column<Xform> = [1.0, 3.0]
        .map(|x| Xform::translation(Vec3::new(x, 0.0, 0.0)))
        .to_vec()
        .into();
    let map = Arc::from([(Id::<Material>::of(id(0)), Id::<Material>::of(id(1)))]);
    let instancer = Instancer::builder(vec![id(2)], xforms)
        .material_maps(MaterialMaps::new(
            vec![Subsets::DEFAULT, 0].into(),
            vec![map],
        ))
        .build()
        .unwrap();
    sync.apply(added(vec![
        (id(0), red),
        (id(1), blue),
        (id(2), Prim::Shape(square(Some(id(0)), [1.0; 3]))),
        (id(3), Prim::Instancer(instancer)),
    ]));
    let scene = sync.scene();
    assert_eq!(scene.instances.len(), 2);
    let color = |x: f32| {
        let k = (0..2)
            .find(|&k| placed(scene, k).iter().any(|p| p[0] == x))
            .unwrap();
        scene.surfaces[scene.instances[k].surface as usize].color
    };
    assert_eq!(color(1.0), [1.0, 0.0, 0.0], "its prototype's own");
    assert_eq!(color(3.0), [0.0, 0.0, 1.0], "mapped");
}

/// Instances differ by their own primvars: each instance's `Cd` stands for
/// its prototype's corners' (the Pixel Surface reads `Cd`), and its `id` is
/// its object id; an Instancer's whole `Cd` colors each of its instances,
/// and instances without an `id` count from 1.
#[test]
fn an_instances_own_color_and_id_are_its_own() {
    let mut sync = Sync::default();
    let surface = material(SURFACE, &[]);
    let row = |x: f64| Xform::translation(Vec3::new(x, 0.0, 0.0));
    let colors = Attribute::tuples(vec![[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]], Meaning::Color);
    let ids = Attribute::tuples(vec![40i64, 41], Meaning::None);
    let own = (kinogaki_render_api::geometry::Attributes::new())
        .with(Domain::Instance, names::CD, colors.unwrap())
        .and_then(|a| a.with(Domain::Instance, names::ID, ids.unwrap()))
        .unwrap();
    let inner = Instancer::builder(vec![id(1)], vec![row(0.0), row(2.0)].into())
        .primvars(own)
        .build()
        .unwrap();
    let whole = Attribute::tuples(vec![[0.0f32, 0.0, 1.0]], Meaning::Color).unwrap();
    let blue = (kinogaki_render_api::geometry::Attributes::new())
        .with(Domain::Detail, names::CD, whole)
        .unwrap();
    let outer = Instancer::builder(vec![id(1)], vec![row(4.0), row(6.0)].into())
        .primvars(blue)
        .build()
        .unwrap();
    sync.apply(added(vec![
        (id(0), surface),
        (id(1), Prim::Shape(square(Some(id(0)), [0.5; 3]))),
        (id(2), Prim::Instancer(inner)),
        (id(3), Prim::Instancer(outer)),
    ]));
    let scene = sync.scene();
    let mut seen: Vec<(u32, Option<[f32; 3]>)> =
        (scene.instances.iter()).map(|i| (i.id, i.tint)).collect();
    seen.sort_by_key(|(id, _)| *id);
    let blue = Some([0.0, 0.0, 1.0]);
    assert_eq!(
        seen,
        [
            (1, blue),
            (2, blue),
            (40, Some([1.0, 0.0, 0.0])),
            (41, Some([0.0, 1.0, 0.0]))
        ]
    );
}

#[test]
fn lights_domes_rims_and_fog_light_the_scene() {
    let mut sync = Sync::default();
    let sun = Light {
        shape: LightShape::Distant { angle: 0.0 },
        intensity: 3.0,
        normalize: true,
        ..Light::default()
    }
    .declared();
    let dome = Light {
        shape: LightShape::Dome {
            image: None,
            ground: Some([0.1, 0.2, 0.3]),
        },
        intensity: 2.0,
        ..Light::default()
    }
    .declared();
    let rim_type = light_types()
        .into_iter()
        .find(|t| t.id == RIM_LIGHT)
        .unwrap();
    let mut rim = Declared::new(RIM_LIGHT, rim_type.schema().values());
    rim.values.insert("side".into(), standard::choice("all"));
    let mut fog = Declared::new(standard::FOG, standard::fog().schema().values());
    fog.values.insert("inscatter".into(), Value::Bool(true));
    let mut hidden = sun.clone();
    hidden.visibility.visible = false;
    let prims = [sun, dome, rim, fog, hidden].map(Prim::Declared);
    let changed = sync.apply(added(
        prims
            .into_iter()
            .enumerate()
            .map(|(k, p)| (id(k as u32), p))
            .collect(),
    ));
    assert!(changed.lights && !changed.meshes && !changed.instances);
    let scene = sync.scene();
    assert_eq!(scene.lights.len(), 1, "the sun; the hidden one is out");
    assert_eq!(scene.lights[0].radiance, [3.0; 3]);
    assert_eq!(
        (scene.ambient.sky, scene.ambient.ground),
        ([2.0; 3], [0.2, 0.4, 0.6])
    );
    assert!(scene.ambient.seen);
    assert_eq!(
        (scene.rims.all, scene.rims.lit),
        ([0.25, 0.2125, 0.15], [0.0; 3])
    );
    assert!(scene.fog.is_some_and(|f| f.lit && f.density == 0.1));
}

#[test]
fn a_removed_shape_leaves_the_scene() {
    let mut sync = Sync::default();
    sync.apply(added(vec![(id(0), Prim::Shape(square(None, [1.0; 3])))]));
    let changed = sync.apply(SceneUpdate {
        removed: vec![id(0)],
        ..SceneUpdate::new()
    });
    assert!(changed.meshes && changed.instances);
    assert!(sync.scene().meshes.is_empty() && sync.scene().instances.is_empty());
}

/// An Instancer a later change makes place itself (through another) places
/// nothing round the cycle, and says so; deep nesting short of one is
/// placed whole.
#[test]
fn a_cycle_of_instancers_places_nothing_and_says_so() {
    let mut sync = Sync::default();
    let instancer = |prototype: PrimId| {
        let places: Column<Xform> = vec![Xform::IDENTITY].into();
        Instancer::builder(vec![prototype], places).build().unwrap()
    };
    // A chain 20 deep: deeper than any cap, placed whole.
    let mut prims = vec![(id(0), Prim::Shape(square(None, [1.0; 3])))];
    prims.extend((1..=20).map(|k| (id(k), Prim::Instancer(instancer(id(k - 1))))));
    sync.apply(added(prims));
    assert_eq!(
        sync.scene().instances.len(),
        1,
        "the shape, through 20 Instancers"
    );
    assert!(sync.problems().is_empty());
    // Instancer 1 now places 3, which places 2, which places 1.
    let change = PrimChange::new(id(1), Dirty::ALL, Prim::Instancer(instancer(id(3))));
    sync.apply(SceneUpdate {
        changed: vec![change],
        ..SceneUpdate::new()
    });
    assert_eq!(
        sync.scene().instances.len(),
        1,
        "the shape alone, named by none now"
    );
    let said = sync.problems();
    assert!(said.iter().any(|d| d.message.contains("cycle")), "{said:?}");
}

/// A Pixel Surface whose color a texture node drives, reading Texture prim
/// `texture`.
fn textured(texture: PrimId) -> Prim {
    use kinogaki_render_api::geometry::image::ImageSource;
    use kinogaki_render_api::material::mtlx::def;
    let description = kinogaki_pixel::description::description();
    let surface = description.node_def(SURFACE).unwrap().clone();
    let mut built = MaterialNetwork::builder();
    let s = built.node(&surface, &Parameters::new()).unwrap();
    let image = [("filtertype".to_owned(), Value::Choice("closest".into()))].into();
    let t = built
        .node(&def("ND_image_color3").unwrap(), &image)
        .unwrap();
    built
        .image(t, "file", ImageSource::path("/t/two.png"))
        .unwrap();
    built.link(t, "out", s, "color").unwrap();
    let mut network = built.build(s).unwrap();
    // What the host does as it sends it: the image's Texture prim.
    network.set_texture(t, "file", Id::of(texture)).unwrap();
    Prim::Material(Material {
        network: Arc::new(network),
        path: Arc::from("/m"),
        double_sided: true,
    })
}

/// A 2 × 1 picture: `left` then `right`, rgba, premultiplied as the host
/// sends colors.
fn picture(left: [f32; 4], right: [f32; 4]) -> Prim {
    use kinogaki_render_api::geometry::image::Encoding;
    encoded(left, right, Encoding::color("ACEScg", true))
}

/// [`picture`] of values `encoding` says what they are.
fn encoded(
    left: [f32; 4],
    right: [f32; 4],
    encoding: kinogaki_render_api::geometry::image::Encoding,
) -> Prim {
    use kinogaki_render_api::geometry::image::{
        Decoding, Encoding, Format, Image, Rect, Sample, TextureImage,
    };
    let format = Format::rgba(Sample::F32, encoding.clone());
    let texels = [left, right].concat();
    let decoding = (encoding != Encoding::Raw).then_some(Decoding::IDENTITY);
    let image = Image::builder(Rect::new([0, 0], [2, 1]).unwrap())
        .plane(format, &texels)
        .build()
        .unwrap();
    Prim::Texture(TextureImage { image, decoding })
}

#[test]
fn a_texture_samples_its_prims_picture_at_each_corners_coordinates() {
    let mut sync = Sync::default();
    let uv = Attribute::tuples(
        vec![[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        Meaning::TextureCoord,
    )
    .unwrap();
    let shape = (square(Some(id(0)), [1.0; 3]).rebuilt())
        .primvar(Domain::Point, names::UV, uv)
        .build()
        .unwrap();
    let half = [0.25, 0.25, 0.25, 0.5];
    let changed = sync.apply(added(vec![
        (id(0), textured(id(2))),
        (id(1), Prim::Shape(shape)),
        (id(2), picture([1.0, 0.0, 0.0, 1.0], half)),
    ]));
    assert!(changed.meshes && changed.surfaces);
    let scene = sync.scene();
    let [picture] = &scene.pictures[..] else {
        panic!("one picture");
    };
    assert_eq!(picture.size, [2, 1]);
    assert_eq!(picture.texels[1], [0.5, 0.5, 0.5, 0.5], "made straight");
    let texture = scene.surfaces[0].texture.expect("its texture");
    assert!(texture.nearest && texture.picture == 0);
    let uvs = scene.meshes[0]
        .uvs
        .as_ref()
        .expect("the corners' coordinates");
    assert_eq!(uvs[..3], [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
    // A new picture in the prim is read again.
    let change = PrimChange::new(
        id(2),
        Dirty::ALL,
        self::picture([0.0, 1.0, 0.0, 1.0], [0.0; 4]),
    );
    let changed = sync.apply(SceneUpdate {
        changed: vec![change],
        ..SceneUpdate::new()
    });
    assert!(changed.surfaces);
    assert_eq!(sync.scene().pictures[0].texels[0], [0.0, 1.0, 0.0, 1.0]);
}

/// A picture of data comes straight (only colors a host premultiplies are
/// divided by their alpha).
#[test]
fn a_texture_of_data_keeps_its_values() {
    use kinogaki_render_api::geometry::image::Encoding;
    let mut sync = Sync::default();
    let half = [0.25, 0.25, 0.25, 0.5];
    sync.apply(added(vec![
        (id(0), textured(id(2))),
        (id(1), Prim::Shape(square(Some(id(0)), [1.0; 3]))),
        (id(2), encoded([1.0; 4], half, Encoding::Raw)),
    ]));
    assert_eq!(sync.scene().pictures[0].texels[1], half);
}

/// glTF's color image with vertex colors: its alpha times the mesh's
/// `Alpha` (opaque where the mesh has none) rides with each corner's tint.
/// A Pixel Surface colored by a glTF color image (texture prim `id(2)`)
/// whose vertex alpha is the mesh's `Alpha`.
fn gltf_material() -> Prim {
    use kinogaki_render_api::geometry::image::ImageSource;
    use kinogaki_render_api::material::mtlx::def;
    let description = kinogaki_pixel::description::description();
    let surface = description.node_def(SURFACE).unwrap().clone();
    let mut built = MaterialNetwork::builder();
    let s = built.node(&surface, &Parameters::new()).unwrap();
    let image = (built.node(
        &def("ND_kinogaki_gltf_colorimage").unwrap(),
        &Parameters::new(),
    ))
    .unwrap();
    let vertex = built
        .node(&def("ND_geomcolor_color4").unwrap(), &Parameters::new())
        .unwrap();
    built
        .image(image, "file", ImageSource::path("/t/two.png"))
        .unwrap();
    built.link(vertex, "out", image, "geomcolor").unwrap();
    built.link(image, "outcolor", s, "color").unwrap();
    let mut network = built.build(s).unwrap();
    network.set_texture(image, "file", Id::of(id(2))).unwrap();
    Prim::Material(Material {
        network: Arc::new(network),
        path: Arc::from("/m"),
        double_sided: true,
    })
}

#[test]
fn a_gltf_color_image_reads_the_vertex_alpha() {
    let material = gltf_material();
    let alpha = Attribute::tuples(vec![0.25f32; 4], Meaning::None).unwrap();
    let faded = (square(Some(id(0)), [0.5; 3]).rebuilt())
        .primvar(Domain::Point, names::ALPHA, alpha)
        .build()
        .unwrap();
    for (shape, expected) in [
        (faded, [0.5, 0.5, 0.5, 0.25]),
        (square(Some(id(0)), [0.5; 3]), [0.5, 0.5, 0.5, 1.0]),
    ] {
        let mut sync = Sync::default();
        sync.apply(added(vec![
            (id(0), material.clone()),
            (id(1), Prim::Shape(shape)),
            (id(2), picture([1.0; 4], [1.0; 4])),
        ]));
        let tints = sync.scene().meshes[0].tints.clone();
        assert_eq!(tints.map(|t| t[0]), Some(expected));
    }
}

/// An instance's own `Alpha` stands for its mesh's vertex alpha, as its own
/// `Cd` stands for its tint; one holding none keeps its mesh's.
#[test]
fn an_instances_own_alpha_is_its_own() {
    let row = |x: f64| Xform::translation(Vec3::new(x, 0.0, 0.0));
    let alphas = Attribute::tuples(vec![0.5f32, 0.75], Meaning::None).unwrap();
    let own = (kinogaki_render_api::geometry::Attributes::new())
        .with(Domain::Instance, names::ALPHA, alphas)
        .unwrap();
    let held = Instancer::builder(vec![id(1)], vec![row(0.0), row(2.0)].into())
        .primvars(own)
        .build()
        .unwrap();
    let bare = Instancer::builder(vec![id(1)], vec![row(4.0)].into())
        .build()
        .unwrap();
    let mut sync = Sync::default();
    sync.apply(added(vec![
        (id(0), gltf_material()),
        (id(1), Prim::Shape(square(Some(id(0)), [0.5; 3]))),
        (id(2), picture([1.0; 4], [1.0; 4])),
        (id(3), Prim::Instancer(held)),
        (id(4), Prim::Instancer(bare)),
    ]));
    let mut seen: Vec<(u32, Option<f32>)> = (sync.scene().instances.iter())
        .map(|i| (i.id, i.alpha))
        .collect();
    seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(seen, [(1, None), (1, Some(0.5)), (2, Some(0.75))]);
}
