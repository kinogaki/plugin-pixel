//! The GPU tests' device, canonical scene and PNG writer.
#![allow(dead_code)]

use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use kinogaki_pixel_render::{
    Aov, AovImage, Camera, Dome, Light, LightShape, Matrix, Mesh, PixelSurface, Projection, Scene,
    Settings,
};
use kinogaki_pixel_render_wgpu::{Renderer, View, Viewport};

/// wgpu's native futures are ready when first polled.
fn ready<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = future.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

/// The default adapter's device, with timestamps when it has them.
pub fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = ready(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("a GPU adapter");
    let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    ready(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: features,
        ..Default::default()
    }))
    .expect("a device")
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    a.map(|v| v / l)
}

/// A frame at `eye` whose -Z looks at `target`, +Y up.
pub fn look_at(eye: [f64; 3], target: [f64; 3]) -> Matrix {
    let back = unit(sub(eye, target));
    let right = unit(cross([0.0, 1.0, 0.0], back));
    let up = cross(back, right);
    let c = |v: [f64; 3], w: f32| [v[0] as f32, v[1] as f32, v[2] as f32, w];
    [c(right, 0.0), c(up, 0.0), c(back, 0.0), c(eye, 1.0)]
}

/// A UV sphere of `radius` at `centre`, smooth normals.
pub fn sphere(centre: [f32; 3], radius: f32, id: u32, material: u32) -> Mesh {
    let (rings, segments) = (32u32, 64u32);
    let mut points = Vec::new();
    for r in 0..=rings {
        let theta = std::f32::consts::PI * r as f32 / rings as f32;
        for s in 0..=segments {
            let phi = 2.0 * std::f32::consts::PI * s as f32 / segments as f32;
            points.push([
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ]);
        }
    }
    let at = |r: u32, s: u32| r * (segments + 1) + s;
    let mut triangles = Vec::new();
    for r in 0..rings {
        for s in 0..segments {
            triangles.push([at(r, s), at(r + 1, s), at(r + 1, s + 1)]);
            triangles.push([at(r, s), at(r + 1, s + 1), at(r, s + 1)]);
        }
    }
    let normals: Vec<[f32; 3]> = triangles
        .iter()
        .flat_map(|t| t.map(|i| points[i as usize]))
        .collect();
    let mut transform = kinogaki_pixel_render::IDENTITY;
    transform[0][0] = radius;
    transform[1][1] = radius;
    transform[2][2] = radius;
    transform[3] = [centre[0], centre[1], centre[2], 1.0];
    Mesh {
        points: Arc::from(points),
        triangles: Arc::from(triangles),
        normals: Some(Arc::from(normals)),
        tints: None,
        transform,
        material,
        id,
        visible: true,
    }
}

/// A `size` × `size` ground plane at y = 0.
pub fn plane(size: f32, id: u32, material: u32) -> Mesh {
    let h = size / 2.0;
    Mesh {
        points: Arc::from(vec![[-h, 0.0, -h], [h, 0.0, -h], [h, 0.0, h], [-h, 0.0, h]]),
        triangles: Arc::from(vec![[0, 2, 1], [0, 3, 2]]),
        normals: None,
        tints: None,
        transform: kinogaki_pixel_render::IDENTITY,
        material,
        id,
        visible: true,
    }
}

/// A one-sided quad of `corners` (counter-clockwise seen from its front),
/// flat shaded.
pub fn quad(corners: [[f32; 3]; 4], id: u32, material: u32) -> Mesh {
    Mesh {
        points: Arc::from(corners.to_vec()),
        triangles: Arc::from(vec![[0, 1, 2], [0, 2, 3]]),
        normals: None,
        tints: None,
        transform: kinogaki_pixel_render::IDENTITY,
        material,
        id,
        visible: true,
    }
}

/// An axis-aligned box from `lo` to `hi`: twelve triangles, closed.
pub fn cuboid(lo: [f32; 3], hi: [f32; 3], id: u32) -> Mesh {
    let corner = |k: usize| {
        [
            if k & 1 == 0 { lo[0] } else { hi[0] },
            if k & 2 == 0 { lo[1] } else { hi[1] },
            if k & 4 == 0 { lo[2] } else { hi[2] },
        ]
    };
    let quads = [
        [0, 2, 3, 1],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 6, 7, 3],
        [0, 4, 6, 2],
        [1, 3, 7, 5],
    ];
    Mesh {
        points: Arc::from((0..8).map(corner).collect::<Vec<_>>()),
        triangles: Arc::from(
            quads
                .iter()
                .flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
                .collect::<Vec<_>>(),
        ),
        normals: None,
        tints: None,
        transform: kinogaki_pixel_render::IDENTITY,
        material: 0,
        id,
        visible: true,
    }
}

/// An orthographic view from `eye` toward `target`, `across` units wide
/// and high, at `size`² view pixels.
pub fn ortho(eye: [f64; 3], target: [f64; 3], across: f32, size: u32) -> View {
    View {
        camera: Camera {
            projection: Projection::Orthographic,
            horizontal_aperture: across * 1000.0,
            vertical_aperture: across * 1000.0,
            clipping_range: [0.1, 1e4],
            transform: look_at(eye, target),
            ..Camera::default()
        },
        width: size,
        height: size,
    }
}

/// A delta distant light shining from `from` toward the origin, its
/// illuminance `intensity`.
pub fn sun(from: [f64; 3], intensity: f32) -> Light {
    Light {
        shape: LightShape::Distant { angle: 0.0 },
        intensity,
        normalize: true,
        transform: look_at(from, [0.0, 0.0, 0.0]),
        ..Light::default()
    }
}

/// `scene` rendered once with `settings` through `view` on a new device.
pub fn render(scene: &Scene, settings: Settings, view: &View) -> Viewport {
    let (device, queue) = device();
    let mut r = Renderer::new(&device, &queue);
    r.set_settings(settings);
    r.set_scene(scene);
    let mut v = Viewport::new(&r, 1);
    r.render(&mut v, view, 0).expect("a frame");
    v
}

/// Film pixel `i`'s first three channels.
pub fn rgb(image: &AovImage, i: usize) -> [f32; 3] {
    let f = image.floats();
    [f[i * 4], f[i * 4 + 1], f[i * 4 + 2]]
}

/// Mesh ids in the canonical scene.
pub const SPHERE: u32 = 1;
/// The plane's id.
pub const PLANE: u32 = 2;

/// The canonical scene: a unit sphere resting on a plane, one rect light
/// above and to the side, a dim dome.
pub fn canonical() -> Scene {
    Scene {
        volumes: Vec::new(),
        meshes: vec![
            sphere([0.0, 1.0, 0.0], 1.0, SPHERE, 0),
            plane(60.0, PLANE, 1),
        ],
        materials: vec![
            PixelSurface {
                color: [0.75, 0.32, 0.12],
                ..PixelSurface::default()
            },
            PixelSurface {
                color: [0.45, 0.5, 0.55],
                ..PixelSurface::default()
            },
        ],
        lights: vec![Light {
            shape: LightShape::Rect {
                width: 1.0,
                height: 1.0,
            },
            intensity: 60.0,
            transform: look_at([4.0, 3.5, 1.0], [0.0, 0.5, 0.0]),
            ..Light::default()
        }],
        dome: Some(Dome {
            above: [0.6 * 0.35, 0.75 * 0.35, 0.35],
            below: [0.6 * 0.35, 0.75 * 0.35, 0.35],
            visible: true,
        }),
        rims: Default::default(),
        fog: None,
    }
}

/// The canonical view: from the front, a little above, at `width`×`height`.
pub fn canonical_view(width: u32, height: u32) -> View {
    View {
        camera: Camera {
            focal_length: 35.0,
            transform: look_at([0.0, 2.4, 7.0], [0.0, 0.8, 0.0]),
            ..Camera::default()
        },
        width,
        height,
    }
}

/// Where the test writes its PNGs.
pub fn out_dir() -> PathBuf {
    let dir = std::env::var_os("PIXEL_RENDER_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../out"));
    std::fs::create_dir_all(&dir).expect("the output directory");
    dir
}

/// ACEScg → linear Rec.709 (Bradford D60 → D65).
const AP1_TO_REC709: [[f32; 3]; 3] = [
    [1.705_051, -0.621_792, -0.083_259],
    [-0.130_256, 1.140_804, -0.010_548],
    [-0.024_003, -0.128_969, 1.152_972],
];

fn srgb(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (e * 255.0 + 0.5) as u8
}

/// An output as 8-bit sRGB for a look (a plain preview transform, not
/// Studio's OCIO view): colors ACEScg → Rec.709 → sRGB, normals ×½+½, depth
/// over its range, ids as flat colors.
pub fn preview(aov: Aov, image: &AovImage) -> Vec<u8> {
    let n = (image.width * image.height) as usize;
    let mut out = Vec::with_capacity(n * 4);
    match aov {
        Aov::Id => {
            for &id in image.ints() {
                let c = if id == u32::MAX {
                    [0, 0, 0]
                } else {
                    let h = id.wrapping_mul(0x9e37_79b9);
                    [(h >> 24) as u8, (h >> 16) as u8, (h >> 8) as u8]
                };
                out.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        Aov::Depth => {
            let f = image.floats();
            let far = f.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
            for &d in f {
                let g = if d > 0.0 {
                    srgb(1.0 - d / far * 0.9)
                } else {
                    0
                };
                out.extend_from_slice(&[g, g, g, 255]);
            }
        }
        Aov::Ao => {
            for &a in image.floats() {
                let g = srgb(a);
                out.extend_from_slice(&[g, g, g, 255]);
            }
        }
        Aov::Normal => {
            for p in image.floats().chunks(4) {
                let c = [0, 1, 2].map(|k| ((p[k] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                out.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        _ => {
            for p in image.floats().chunks(4) {
                let m = AP1_TO_REC709;
                let c = m.map(|row| srgb(row[0] * p[0] + row[1] * p[1] + row[2] * p[2]));
                out.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
    }
    out
}

/// Writes `rgba` as `name`.png in the output directory.
pub fn write_png(name: &str, width: u32, height: u32, rgba: Vec<u8>) -> PathBuf {
    let bytes = png(width, height, &rgba);
    let path = out_dir().join(format!("{name}.png"));
    std::fs::write(&path, bytes).expect("the PNG written");
    path
}

/// `rgba` (8 bits a channel, rows top down) as a PNG: no filtering, and
/// zlib with stored (uncompressed) deflate blocks.
pub fn png(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let row = width as usize * 4;
    assert_eq!(rgba.len(), row * height as usize, "{width}×{height} RGBA8");
    // Each row with filter type 0 (none) in front.
    let mut raw = Vec::with_capacity((row + 1) * height as usize);
    for line in rgba.chunks_exact(row.max(1)).take(height as usize) {
        raw.push(0);
        raw.extend_from_slice(line);
    }
    let mut zlib = vec![0x78, 0x01];
    let mut blocks = raw.chunks(0xffff).peekable();
    if blocks.peek().is_none() {
        zlib.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
    }
    while let Some(block) = blocks.next() {
        let last = u8::from(blocks.peek().is_none());
        let len = block.len() as u16;
        zlib.push(last);
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    // 8 bits, RGBA, deflate, adaptive filtering, no interlace.
    header.extend_from_slice(&[8, 6, 0, 0, 0]);

    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    for (kind, data) in [(b"IHDR", &header), (b"IDAT", &zlib), (b"IEND", &Vec::new())] {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let start = out.len();
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        let crc = crc32(&out[start..]);
        out.extend_from_slice(&crc.to_be_bytes());
    }
    out
}

/// CRC-32 (ISO 3309, reflected, polynomial 0xedb88320), bit by bit.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Adler-32, zlib's checksum of the uncompressed data.
fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in bytes.chunks(5552) {
        for &x in chunk {
            a += u32::from(x);
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}
