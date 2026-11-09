//! Portals and geometry lights as Pixel lights them, on the floor of
//! `lights.rs`: a dome a portal names lights the floor's ambient through
//! the portal alone; a geometry light lights as a sphere of its mesh's area
//! at its centre, its own closed mesh casting no shadow of it.
use super::lights::{BELOW, SIDE, down, shown};
use super::support::id;
use kinogaki_render_api::aov::standard as aovs;
use kinogaki_render_api::geometry::{Column, Sampled, Vec3};
use kinogaki_render_api::standard::{DomeFormat, Light, LightShape, MaterialSync, Pole};
use kinogaki_render_api::{Id, Material, Members, Prim, Shape, Subsets, Topology};
use std::f32::consts::PI;

/// A pixel far out on the floor (x 10.5).
const FAR: (u32, u32) = (SIDE - 2, 12);

fn dome() -> Light {
    Light {
        shape: LightShape::Dome {
            image: None,
            format: DomeFormat::Automatic,
            pole: Pole::Scene,
            ground: None,
        },
        ..Light::default()
    }
}

/// `light` as a prim whose object row `row` names the prims of slots
/// `slots`.
fn naming(light: &Light, row: &str, slots: &[u32]) -> Prim {
    let mut prim = light.declared();
    let named = Members::new(slots.iter().map(|&s| id(s)));
    prim.members.insert(row.to_owned(), named);
    Prim::Declared(prim)
}

#[test]
#[ignore = "gpu"]
fn a_portal_lets_its_domes_ambient_through_alone() {
    let ambient = |prims: Vec<Prim>| shown(prims, false, 0.5, "ambient", &[BELOW, FAR]);
    let open = ambient(vec![Prim::Declared(dome().declared())]);
    // A 2 × 2 portal 1 above the pixel below, its light passing down; the
    // dome is prim 10.
    let portal = Light {
        shape: LightShape::Portal {
            width: 2.0,
            height: 2.0,
            dome: None,
        },
        xform: Sampled::still(down(Vec3::new(0.5, 1.0, 0.5))),
        ..Light::default()
    };
    let through = ambient(vec![
        Prim::Declared(dome().declared()),
        naming(&portal, "dome", &[10]),
    ]);
    println!("open {open:?}, through {through:?}");
    assert!(
        (open[0][1] - open[1][1]).abs() < 1e-3,
        "an open dome lights all round"
    );
    assert!(
        through[0][1] > 0.1 && through[0][1] < 0.9 * open[0][1],
        "{through:?}"
    );
    assert!(through[1][1] < 0.05 * open[1][1], "{through:?}");
}

/// A cube of edge 1 centred `HIGH` (4) above the pixel below.
fn cube() -> Shape {
    let (c, h) = ([0.5f32, 4.0, 0.5], 0.5f32);
    let points: Vec<[f32; 3]> = (0..8)
        .map(|i| {
            let s = |bit: i32| if i & bit == 0 { -h } else { h };
            [c[0] + s(1), c[1] + s(2), c[2] + s(4)]
        })
        .collect();
    let quads = [
        [0, 2, 3, 1],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 4, 6, 2],
        [1, 3, 7, 5],
    ];
    let triangles: Vec<[u32; 3]> = quads
        .iter()
        .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
        .collect();
    let count = triangles.len();
    Shape::builder(Column::from(points), Topology::Triangles(triangles.into()))
        .subsets(Subsets::new(
            vec![0u16; count].into(),
            vec![Id::<Material>::of(id(0))],
        ))
        .build()
        .unwrap()
}

#[test]
#[ignore = "gpu"]
fn a_geometry_light_lights_as_a_sphere_of_its_mesh_and_its_mesh_casts_no_shadow() {
    let lamp = Light {
        shape: LightShape::Geometry {
            geometry: None,
            sync: MaterialSync::NoMaterialResponse,
        },
        intensity: 2.0,
        ..Light::default()
    };
    // The cube is prim 10, the light 11.
    let lit = shown(
        vec![Prim::Shape(cube()), naming(&lamp, "geometry", &[10])],
        false,
        0.5,
        aovs::DIRECT,
        &[BELOW],
    );
    // E / π: the radiance × a quarter of its area (6), over d² = 16, over π.
    let expected = 2.0 * 6.0 / 4.0 / 16.0 / PI;
    println!("below {:?}, expected {expected}", lit[0]);
    assert!((lit[0][1] - expected).abs() <= 1e-3 * expected, "{lit:?}");
}
