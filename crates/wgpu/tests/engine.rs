//! Pixel Render as an engine: made through `Pixel`, fed the render
//! interface's prims, it renders a view when something changed, wakes once
//! per frame and never writes an image a host shows; final frames read
//! their outputs. `cargo test -- --ignored` runs it.

mod support;

use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use std::time::Duration;

use kinogaki_pixel_render::material;
use kinogaki_pixel_render::rim::{Rim, RimSide};
use kinogaki_pixel_render_wgpu::Pixel;
use kinogaki_render_api::aov::standard as aov;
use kinogaki_render_api::display::{Engine, Options};
use kinogaki_render_api::standard::{Fog, Light, Shape};
use kinogaki_render_api::{
    Aov, AovDesc, AovFormat, Camera, Material, MaterialNetwork, Mesh, Prim, PrimId, SceneUpdate,
    ViewCamera, ViewDesc, ViewId, ViewMode, Wake,
};
use support::{device, look_at};

fn id(slot: u32) -> PrimId {
    PrimId {
        slot,
        generation: 0,
    }
}

/// A wake that sends on a channel the test waits on.
fn wake() -> (Wake, Receiver<()>) {
    let (send, receive) = channel();
    (
        Arc::new(move || {
            let _ = send.send(());
        }),
        receive,
    )
}

fn waited(woken: &Receiver<()>) {
    woken.recv_timeout(Duration::from_secs(20)).expect("a wake");
}

/// A quad facing +Z, 2 units across, of material slot 0.
fn quad(material: PrimId) -> Prim {
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
        materials: Arc::from(vec![material]),
        attributes: Arc::from([]),
        transform: kinogaki_render_api::IDENTITY,
        visible: true,
        id: 3,
        path: Arc::from("/quad"),
    })
}

fn scene(network: MaterialNetwork) -> SceneUpdate {
    let mut update = SceneUpdate::new();
    update.added.push((
        id(0),
        Prim::Material(Material {
            network: Arc::new(network),
            path: Arc::from("/m"),
        }),
    ));
    update.added.push((id(1), quad(id(0))));
    update.added.push((
        id(2),
        Prim::Declared(
            Light {
                shape: Shape::Dome {
                    texture: None,
                    ground: None,
                },
                intensity: 1.0,
                ..Light::default()
            }
            .declared(),
        ),
    ));
    update
}

fn view(z: f32) -> ViewDesc {
    ViewDesc {
        camera: ViewCamera::Free(Camera {
            transform: look_at([0.0, 0.0, z as f64], [0.0, 0.0, 0.0]),
            ..Camera::default()
        }),
        width: 160,
        height: 120,
        mode: ViewMode::Interactive,
    }
}

/// The centre pixel of a frame's pixels, as floats.
fn centre(frame: &kinogaki_render_api::Frame) -> [f32; 4] {
    let at = ((frame.height / 2 * frame.width + frame.width / 2) * 4) as usize;
    std::array::from_fn(|k| kinogaki_pixel_render_wgpu::f16_to_f32(frame.pixels[at + k]))
}

#[test]
#[ignore = "needs a GPU"]
fn frames_come_when_something_changed_and_shown_images_stay() {
    let (device, queue) = device();
    let (wake, woken) = wake();
    let options = Options {
        cache: None,
        read_pixels: true,
    };
    let (mut engine, mut display) = Pixel
        .make(&device, &queue, wake, &options)
        .expect("an engine");
    let v = ViewId(0);
    engine.update(scene(material::network([0.5, 0.25, 0.125], "Cd")));
    engine.view(v, view(4.0));
    waited(&woken);
    let first = engine.latest(v).expect("a frame");
    assert_eq!((first.width, first.height), (160, 120));
    assert_eq!((first.stats.samples, first.stats.target), (1, 1));
    // The quad lit by the dome alone: its color × the shaded ambient light.
    let c = centre(&first);
    assert!(c[0] > 0.0 && c[0] > c[1] && c[1] > c[2], "{c:?}");
    // Nothing changed: no frame, no wake.
    engine.view(v, view(4.0));
    assert!(woken.recv_timeout(Duration::from_millis(300)).is_err());

    // A shown frame's image is never written while it is shown.
    let shown = display.acquire(&first).expect("its image");
    assert_eq!(shown.filter(), wgpu::FilterMode::Nearest);
    for k in 0..4 {
        engine.view(v, view(4.5 + k as f32));
        waited(&woken);
        let newest = engine.latest(v).unwrap();
        assert!(newest.generation > first.generation);
        assert_ne!(newest.image, first.image, "the shown image was written");
    }
    drop(shown);
    // A material edit reaches the next frame.
    let mut edit = SceneUpdate::new();
    edit.changed.push(kinogaki_render_api::Change::new(
        id(0),
        kinogaki_render_api::Dirty::PARAMS,
        Prim::Material(Material {
            network: Arc::new(material::network([0.0, 0.5, 0.0], "Cd")),
            path: Arc::from("/m"),
        }),
    ));
    engine.update(edit);
    waited(&woken);
    let green = centre(&engine.latest(v).unwrap());
    assert!(green[1] > green[0] && green[0] == 0.0, "{green:?}");
}

#[test]
#[ignore = "needs a GPU"]
fn fog_and_rim_lights_come_as_prims() {
    let (device, queue) = device();
    let (wake, woken) = wake();
    let options = Options {
        cache: None,
        read_pixels: true,
    };
    let (mut engine, _display) = Pixel.make(&device, &queue, wake, &options).unwrap();
    let v = ViewId(3);
    engine.update(scene(material::network([0.5, 0.5, 0.5], "Cd")));
    engine.view(v, view(4.0));
    waited(&woken);
    let clear = centre(&engine.latest(v).unwrap());
    // Thick fog: the quad 4 units off is all fog.
    let red = [0.8, 0.1, 0.05];
    let mut add = SceneUpdate::new();
    add.added.push((
        id(7),
        Prim::Declared(
            Fog {
                density: 10.0,
                color: red,
                falloff: 0.0,
                ..Fog::default()
            }
            .declared(),
        ),
    ));
    engine.update(add);
    waited(&woken);
    let fogged = centre(&engine.latest(v).unwrap());
    for k in 0..3 {
        assert!((fogged[k] - red[k]).abs() < 2e-3, "{fogged:?}");
    }
    // Gone: the quad again.
    let mut gone = SceneUpdate::new();
    gone.removed.push(id(7));
    engine.update(gone);
    waited(&woken);
    assert_eq!(centre(&engine.latest(v).unwrap()), clear);
    // A rim light all round: the quad faces the camera, so its middle
    // takes no rim; a frame comes all the same.
    let mut rim = SceneUpdate::new();
    rim.added.push((
        id(8),
        Prim::Declared(
            Rim {
                side: RimSide::All,
                intensity: 5.0,
                ..Rim::default()
            }
            .declared(),
        ),
    ));
    engine.update(rim);
    waited(&woken);
    assert_eq!(centre(&engine.latest(v).unwrap()), clear);
}

#[test]
#[ignore = "needs a GPU"]
fn final_frames_read_their_outputs_at_the_view_size() {
    let (device, queue) = device();
    let (wake, woken) = wake();
    let (mut engine, _display) = Pixel
        .make(&device, &queue, wake, &Options::default())
        .unwrap();
    // Another engine's material renders as the default Pixel Surface.
    let foreign = MaterialNetwork {
        engine: Arc::from("spectre"),
        ..MaterialNetwork::default()
    };
    engine.update(scene(foreign));
    let v = ViewId(16);
    engine.view(
        v,
        ViewDesc {
            mode: ViewMode::Final,
            ..view(4.0)
        },
    );
    let wanted: Vec<AovDesc> = [
        aov::BEAUTY,
        aov::ALBEDO,
        aov::OCCLUSION,
        aov::OBJECT_ID,
        aov::POSITION,
    ]
    .into_iter()
    .map(|id| AovDesc {
        aov: aov::get(id).unwrap(),
        format: AovFormat::F32,
    })
    .collect();
    let pending = engine.finish(v, &wanted);
    waited(&woken);
    let outputs = pending.take().expect("the outputs");
    assert_eq!((outputs.width, outputs.height), (160, 120));
    // Position is not Pixel Render's: left out.
    let kinds: Vec<_> = outputs.planes.iter().map(|(d, _)| &d.aov).collect();
    assert_eq!(
        &kinds[..],
        &wanted[..4].iter().map(|d| &d.aov).collect::<Vec<_>>()[..]
    );
    let pixels = 160 * 120;
    for (desc, values) in &outputs.planes {
        assert_eq!(values.len(), pixels * desc.aov.channels(), "{:?}", desc.aov);
    }
    let centre = (60 * 160 + 80) as usize;
    let albedo = &outputs.planes[1].1;
    assert!(
        (albedo[centre * 3] - 0.8).abs() < 1e-3,
        "the default surface"
    );
    assert_eq!(outputs.planes[3].1[centre].to_bits(), 3, "the quad's id");
    // Each film pixel covers a 4×4 block: x 80 and 81 are one pixel.
    let beauty = &outputs.planes[0].1;
    let at = |i: usize| &beauty[i * 4..i * 4 + 4];
    assert_eq!(at(centre), at(centre + 1));
}

/// A compositing host's film: the beauty at film size from the first frame,
/// the outputs it asks for from the next, each texel what a final frame
/// repeats over its pixel-size block.
#[test]
#[ignore = "needs a GPU"]
fn the_film_is_the_final_outputs_decimated_by_block() {
    use kinogaki_pixel_render::AovPixels;
    let (device, queue) = device();
    let (wake, woken) = wake();
    let (mut engine, mut display) = Pixel
        .make(&device, &queue, wake, &Options::default())
        .unwrap();
    engine.update(scene(material::network([0.5, 0.25, 0.125], "Cd")));
    let v = ViewId(2);
    engine.view(v, view(4.0));
    waited(&woken);
    let asked: Vec<Aov> = [aov::ALBEDO, aov::DEPTH, aov::OBJECT_ID]
        .map(|id| aov::get(id).unwrap())
        .to_vec();
    let first = engine.latest(v).expect("a frame");
    assert_eq!(first.block, 4, "the default pixel size");
    let film = display.acquire_film(&first, &asked).expect("its film");
    assert_eq!(film.size(), [40, 30]);
    assert_eq!(film.block(), 4);
    let kinds = |f: &kinogaki_render_api::display::ShownFilm| -> Vec<Aov> {
        f.images().iter().map(|i| i.aov.clone()).collect()
    };
    assert_eq!(
        kinds(&film),
        [aov::beauty()],
        "outputs come with the next frame"
    );
    drop(film);
    // Asking rendered the next frame, which keeps them.
    waited(&woken);
    let second = engine.latest(v).expect("a frame");
    let film = display.acquire_film(&second, &asked).expect("its film");
    assert_eq!(kinds(&film), [vec![aov::beauty()], asked.clone()].concat());
    let read: Vec<_> = film
        .images()
        .iter()
        .map(|i| {
            let image = kinogaki_pixel_render_wgpu::read_texture(&device, &queue, i.view.texture());
            (i.aov.clone(), image)
        })
        .collect();
    let all: Vec<AovDesc> = [aov::beauty()]
        .into_iter()
        .chain(asked)
        .map(|aov| AovDesc {
            aov,
            format: AovFormat::F32,
        })
        .collect();
    let pending = engine.finish(v, &all);
    waited(&woken);
    let outputs = pending.take().expect("the outputs");
    assert_eq!(outputs.block, 4);
    for ((kind, image), (desc, plane)) in read.iter().zip(&outputs.planes) {
        assert_eq!(*kind, desc.aov);
        let keep = kind.channels();
        let stride = image.channels as usize;
        let mut checked = 0;
        for j in 0..image.height {
            for i in 0..image.width {
                let (x, y) = (i * outputs.block, j * outputs.block);
                if x >= outputs.width || y >= outputs.height {
                    continue;
                }
                let texel = (j * image.width + i) as usize;
                let pixel = (y * outputs.width + x) as usize;
                for k in 0..keep {
                    let film = match &image.pixels {
                        AovPixels::F32(v) => v[texel * stride + k].to_bits(),
                        AovPixels::U32(v) => v[texel * stride + k],
                    };
                    assert_eq!(
                        film,
                        plane[pixel * keep + k].to_bits(),
                        "{kind:?} at film pixel ({i}, {j}) channel {k}"
                    );
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 40 * 30, "{kind:?}");
    }
}
