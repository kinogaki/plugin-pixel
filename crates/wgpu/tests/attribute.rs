//! A Pixel Surface's color attribute on the GPU: a quad whose two triangles
//! carry different primitive `Cd`s, sent as the render interface's prims
//! through the scene store, renders two albedos. `cargo test -- --ignored`
//! runs it.

mod support;

use std::sync::Arc;

use kinogaki_pixel_render::{Aov, Camera, Scene, Settings, Store, material};
use kinogaki_pixel_render_wgpu::{Renderer, View, Viewport};
use kinogaki_render_api::{
    AttributeDomain, Material, Mesh, MeshAttribute, Prim, PrimId, SceneUpdate,
};
use support::{device, look_at};

fn id(slot: u32) -> PrimId {
    PrimId {
        slot,
        generation: 0,
    }
}

const RED: [f32; 3] = [0.8, 0.1, 0.1];
const BLUE: [f32; 3] = [0.1, 0.2, 0.7];
const COLOR: [f32; 3] = [0.5, 1.0, 1.0];

/// A quad facing +Z, 2 units across: triangle 0 is its lower right, 1 its
/// upper left, each with its own `Cd`.
fn quad() -> Prim {
    Prim::Mesh(Mesh {
        points: Arc::from(vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ]),
        triangles: Arc::from(vec![[0, 1, 2], [0, 2, 3]]),
        normals: None,
        uvs: None,
        material_slots: Arc::from(vec![0u16, 0]),
        materials: Arc::from(vec![id(0)]),
        attributes: Arc::from(vec![MeshAttribute {
            name: Arc::from("Cd"),
            domain: AttributeDomain::Triangle,
            components: 3,
            color: true,
            integer: false,
            values: [RED, BLUE].concat().into(),
        }]),
        transform: kinogaki_render_api::IDENTITY,
        visible: true,
        id: 1,
        path: Arc::from("/quad"),
    })
}

#[test]
#[ignore = "needs a GPU"]
fn primitives_with_different_cd_render_different_albedos() {
    let mut store = Store::new();
    let mut update = SceneUpdate::new();
    update.added.push((
        id(0),
        Prim::Material(Material {
            network: Arc::new(material::network(COLOR, "Cd")),
            path: Arc::from("/m"),
        }),
    ));
    update.added.push((id(1), quad()));
    store.apply(update);
    let scene = Scene {
        meshes: store.meshes(),
        materials: store.materials(),
        ..Scene::default()
    };
    let (device, queue) = device();
    let mut r = Renderer::new(&device, &queue);
    r.set_settings(Settings::default());
    r.set_scene(&scene);
    let mut v = Viewport::new(&r, 1);
    let view = View {
        camera: Camera {
            transform: look_at([0.0, 0.0, 4.0], [0.0, 0.0, 0.0]),
            ..Camera::default()
        },
        width: 160,
        height: 120,
    };
    r.render(&mut v, &view, 0).expect("a frame");
    let film = v.film().expect("a film");
    let albedo = v.read_aov(Aov::Albedo).expect("albedo");
    let at = |x: f32, y: f32| -> [f32; 3] {
        // Film pixels from the centre, about 14 a unit at this distance.
        let px = (film.size[0] as f32 / 2.0 + x * 13.0) as u32;
        let py = (film.size[1] as f32 / 2.0 - y * 13.0) as u32;
        let i = ((py * film.size[0] + px) * 4) as usize;
        std::array::from_fn(|k| albedo.floats()[i + k])
    };
    let close = |a: [f32; 3], b: [f32; 3]| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-3);
    let times = |c: [f32; 3]| std::array::from_fn(|k| COLOR[k] * c[k]);
    let (lower_right, upper_left) = (at(0.5, -0.5), at(-0.5, 0.5));
    assert!(close(lower_right, times(RED)), "{lower_right:?}");
    assert!(close(upper_left, times(BLUE)), "{upper_left:?}");
}
