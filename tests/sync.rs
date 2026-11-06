//! The host's prims as Pixel's scene: shapes as meshes by material in
//! their own space, placed by instances (a move remakes the instances alone), volumes apart, tints and normals per corner,
//! lights, domes, rims and fog, and what each batch says changed.
use kinogaki_pixel::lights::{RIM_LIGHT, light_types};
use kinogaki_pixel::nodes::{SURFACE, VOLUME};
use kinogaki_pixel::sync::Sync;
use kinogaki_render_api::geometry::{
    Attribute, Column, Domain, Meaning, Sampled, Vec3, Xform, names,
};
use kinogaki_render_api::standard::{self, Light, Shape as LightShape};
use kinogaki_render_api::value::{Parameters, Value};
use kinogaki_render_api::{
    Change, Declared, Dirty, Id, Instancer, Material, MaterialNetwork, Prim, PrimId, SceneUpdate,
    Shape, Subsets, Topology,
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

/// Instance `k`'s corners in world space.
fn placed(scene: &kinogaki_pixel::render::Scene, k: usize) -> Vec<[f32; 3]> {
    let i = &scene.instances[k];
    let mesh = &scene.meshes[i.mesh as usize];
    let at = |p: [f32; 3]| {
        i.to_world
            .map(|r| r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3])
    };
    mesh.corners.iter().map(|&p| at(p)).collect()
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
    assert_eq!(mesh.corners.len(), 6);
    assert!(mesh.corners.iter().all(|p| p[2] == 0.0), "its own space");
    assert!(placed(scene, 0).iter().all(|p| p[2] == 5.0));
    assert_eq!(
        mesh.tints.as_deref().map(|t| t[0]),
        Some([0.5; 3]),
        "Cd, the default attribute"
    );
    assert_eq!(
        scene.surfaces[instance.surface as usize].color,
        [1.0, 0.0, 0.0]
    );
    assert_eq!(instance.id, 2, "its prim's slot, plus one");
}

/// A move remakes the instances alone: the mesh is the one made before. A
/// new point remakes it.
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
        changed: vec![Change::new(
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
        changed: vec![Change::new(id(0), Dirty::POINTS, Prim::Shape(shape(2.0)))],
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
        changed: vec![Change::new(id(0), Dirty::PARAMS, blue)],
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
        changed: vec![Change::new(id(0), Dirty::PARAMS, denser)],
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
    let mut rim = Declared::new(RIM_LIGHT, rim_type.defaults());
    rim.values.insert("side".into(), standard::choice("all"));
    let mut fog = Declared::new(standard::FOG, standard::fog().defaults());
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
    let change = Change::new(id(1), Dirty::ALL, Prim::Instancer(instancer(id(3))));
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
